import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import cv2
from PIL import Image, ImageTk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.interpolate import make_interp_spline

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang TC', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class UniversalPhysicsTrackerApp:
    def __init__(self, window):
        self.window = window
        self.window.title("【科研級運動追蹤分析工具】")
        self.window.geometry("1480x920")
        
        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except Exception:
            pass
            
        self.style.configure(
            "White.TButton", 
            background="#ffffff", 
            foreground="#000000", 
            font=("Microsoft JhengHei", 9, "bold"),
            padding=(6, 3),
            relief="raised"
        )
        self.style.map("White.TButton",
            background=[('pressed', '#e0e0e0'), ('active', '#f5f5f5'), ('disabled', '#f0f0f0')],
            foreground=[('disabled', '#888888')]
        )
        
        self.video_path = None
        self.cap = None
        self.tracker = None
        self.tracking = False
        self.selecting = False
        self.bbox = None
        self.first_frame = None
        self.frame_data = []
        self.fps = 30.0
        self.frame_count = 0
        
        self.origin_x = None
        self.origin_y = None
        
        self.lost_frames_limit = 30
        self.lost_frames_counter = 0
        
        self.calibrating = False
        self.scale_points = []
        self.pixel_per_unit = None  
        self.unit_name = "px"       
        
        # 框選
        self.init_x, self.init_y = 0, 0
        self.cur_x, self.cur_y = 0, 0
        self.rect = None
        
        # 控制變數
        self.var_enable_skipping = tk.BooleanVar(value=True)
        self.var_smooth_chart = tk.BooleanVar(value=True)
        
        # 嚴謹物理量定義清單 (鍵值: (中文名稱, 單位格式, 繪圖線條顏色))
        self.metric_defs = {
            "Coord_X": ("X 軸座標 (X)", "{unit}", "blue"),
            "Coord_Y": ("Y 軸座標 (Y)", "{unit}", "navy"),
            "Disp_X": ("X 軸位移分量 (ΔX)", "{unit}", "deepskyblue"),
            "Disp_Y": ("Y 軸位移分量 (ΔY)", "{unit}", "steelblue"),
            "Disp_Total": ("直線位移量 (|Δr|)", "{unit}", "teal"),
            "Path_Length": ("累計路徑長 (濾除≤5px)", "{unit}", "saddlebrown"),
            "Vel_X": ("X 軸速度分量 (Vx)", "{unit}/s", "forestgreen"),
            "Vel_Y": ("Y 軸速度分量 (Vy)", "{unit}/s", "darkgreen"),
            "Speed": ("瞬時速率 (|v|)", "{unit}/s", "purple"),
            "Accel_X": ("X 軸加速度 (ax)", "{unit}/s²", "firebrick"),
            "Accel_Y": ("Y 軸加速度 (ay)", "{unit}/s²", "indianred"),
            "Accel_Mag": ("瞬時加速度大小 (|a|)", "{unit}/s²", "crimson"),
            "Jerk_Mag": ("瞬時急動度 (|J|)", "{unit}/s³", "darkorange")
        }
        
        self.metric_vars = {}
        self.chart_cards = {}  # 儲存獨立圖表卡片
        self.setup_ui()
        
    def setup_ui(self):
        # 左側面板
        left_frame = ttk.Frame(self.window, padding=8)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(side=tk.TOP, fill=tk.X, pady=4)
        
        self.btn_open = ttk.Button(btn_frame, text="1. 開啟影片", command=self.open_video, style="White.TButton")
        self.btn_open.pack(side=tk.LEFT, padx=2)
        
        self.btn_scale = ttk.Button(btn_frame, text="📐 設定比例尺", command=self.start_calibration, state=tk.DISABLED, style="White.TButton")
        self.btn_scale.pack(side=tk.LEFT, padx=2)
        
        self.btn_select = ttk.Button(btn_frame, text="2. 框選目標", command=self.start_selection, state=tk.DISABLED, style="White.TButton")
        self.btn_select.pack(side=tk.LEFT, padx=2)
        
        self.btn_start = ttk.Button(btn_frame, text="3. 播放追蹤", command=self.start_tracking, state=tk.DISABLED, style="White.TButton")
        self.btn_start.pack(side=tk.LEFT, padx=2)
        
        self.btn_fast_start = ttk.Button(btn_frame, text="⚡ 快速出結果", command=self.start_fast_tracking, state=tk.DISABLED, style="White.TButton")
        self.btn_fast_start.pack(side=tk.LEFT, padx=2)
        
        self.btn_export = ttk.Button(btn_frame, text="💾 導出 CSV (自選欄位)", command=self.open_export_dialog, state=tk.DISABLED, style="White.TButton")
        self.btn_export.pack(side=tk.RIGHT, padx=2)
        
        # 播放速度與取樣控制列
        speed_frame = ttk.LabelFrame(left_frame, text="⏩ 播放與取樣控制", padding=6)
        speed_frame.pack(side=tk.TOP, fill=tk.X, pady=4)
        
        slider_row = ttk.Frame(speed_frame)
        slider_row.pack(fill=tk.X, pady=2)
        
        self.speed_slider = tk.Scale(
            slider_row, from_=0.1, to=4.0, resolution=0.1, orient=tk.HORIZONTAL,
            length=230, highlightthickness=0, command=self.on_speed_change, bg="#ffffff"
        )
        self.speed_slider.set(1.0)
        self.speed_slider.pack(side=tk.LEFT, padx=4)
        
        self.lbl_speed_val = ttk.Label(slider_row, text="1.0x (原速)", font=("Microsoft JhengHei", 9))
        self.lbl_speed_val.pack(side=tk.LEFT, padx=4)
        
        btn_reset_speed = ttk.Button(slider_row, text="重設 1.0x", command=lambda: self.speed_slider.set(1.0), style="White.TButton")
        btn_reset_speed.pack(side=tk.LEFT, padx=4)
        
        skip_row = ttk.Frame(speed_frame)
        skip_row.pack(fill=tk.X, pady=2)
        
        chk_skip = ttk.Checkbutton(
            skip_row, text="啟用跳幀加速 (倍速時生效)", variable=self.var_enable_skipping, 
            command=self.update_skip_prediction
        )
        chk_skip.pack(side=tk.LEFT, padx=4)
        
        self.lbl_prediction = ttk.Label(
            skip_row, text="[預測：每 1 影格計算一次 (逐格解算)]", 
            font=("Microsoft JhengHei", 8, "bold"), foreground="teal"
        )
        self.lbl_prediction.pack(side=tk.LEFT, padx=6)
        
        # 影片播放畫布
        self.canvas = tk.Canvas(left_frame, width=640, height=480, bg="black", highlightthickness=1, highlightbackground="#cccccc")
        self.canvas.pack(side=tk.TOP, pady=6)
        
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
        
        self.lbl_scale_info = ttk.Label(left_frame, text="比例尺：未設定 (以像素 px 為基準)", font=("Microsoft JhengHei", 9, "bold"), foreground="purple")
        self.lbl_scale_info.pack(side=tk.TOP, anchor=tk.W, pady=1)
        
        # 底部狀態列 + 左下角資訊按鈕
        bottom_bar = ttk.Frame(left_frame)
        bottom_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=2)
        
        # 左下角作者資訊小按鈕
        btn_info = ttk.Button(bottom_bar, text="ℹ️ 軟體資訊", command=self.show_about_info, style="White.TButton")
        btn_info.pack(side=tk.LEFT, padx=2)
        
        self.lbl_status = ttk.Label(bottom_bar, text="狀態：請先開啟影片", font=("Microsoft JhengHei", 9), foreground="blue")
        self.lbl_status.pack(side=tk.LEFT, padx=8)

        # ==========================================
        # 右側面板：勾選項 + 獨立圖表卡片可滾動容器
        # ==========================================
        self.right_frame = ttk.Frame(self.window, padding=8)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        checkbox_panel = ttk.LabelFrame(self.right_frame, text="🔬 勾選顯示的物理量 (即時繪圖 / 獨立檢視)", padding=6)
        checkbox_panel.pack(side=tk.TOP, fill=tk.X, padx=4, pady=2)
        
        control_subrow = ttk.Frame(checkbox_panel)
        control_subrow.pack(side=tk.TOP, fill=tk.X, pady=1)
        
        chk_smooth = ttk.Checkbutton(
            control_subrow, text="🌊 啟用圖表曲線圓滑化 (三次樣條平滑插值)", 
            variable=self.var_smooth_chart, command=self.update_all_charts
        )
        chk_smooth.pack(side=tk.LEFT, padx=4)
        
        ttk.Separator(checkbox_panel, orient="horizontal").pack(fill=tk.X, pady=3)
        
        metrics_grid = ttk.Frame(checkbox_panel)
        metrics_grid.pack(side=tk.TOP, fill=tk.X)
        
        default_checked = ["Coord_X", "Coord_Y", "Speed", "Accel_Mag", "Path_Length"]
        col_idx, row_idx = 0, 0
        for key, (label_text, _, _) in self.metric_defs.items():
            var = tk.BooleanVar(value=(key in default_checked))
            self.metric_vars[key] = var
            chk = ttk.Checkbutton(
                metrics_grid, text=label_text, variable=var,
                command=self.rebuild_chart_container
            )
            chk.grid(row=row_idx, column=col_idx, sticky="w", padx=4, pady=1)
            col_idx += 1
            if col_idx >= 4:
                col_idx = 0
                row_idx += 1

        # 滾動圖表容器 (改用個別卡片架構，徹底杜絕子圖覆蓋)
        chart_scroll_frame = ttk.Frame(self.right_frame)
        chart_scroll_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=4, pady=4)
        
        self.plot_canvas = tk.Canvas(chart_scroll_frame, bg="#f5f5f5", highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(chart_scroll_frame, orient="vertical", command=self.plot_canvas.yview)
        self.scrollable_inner_frame = ttk.Frame(self.plot_canvas)
        
        self.scrollable_inner_frame.bind(
            "<Configure>",
            lambda e: self.plot_canvas.configure(scrollregion=self.plot_canvas.bbox("all"))
        )
        self.plot_canvas.create_window((0, 0), window=self.scrollable_inner_frame, anchor="nw")
        self.plot_canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.plot_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.plot_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        self.rebuild_chart_container()

    def show_about_info(self):
        info_text = (
            "【科研級運動追蹤分析工具】\n\n"
            "製作人：黃晨揖\n"
            "製作公司：小揖資訊服務\n"
            "開發語言：Python 3.13\n"
            "核心演算法：OpenCV CSRT 高可靠度通道空間追蹤\n"
            "版本：v3.5 科研專用版"
        )
        messagebox.showinfo("關於本軟體", info_text)

    def _on_mousewheel(self, event):
        if event.widget.winfo_toplevel() == self.window:
            self.plot_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def on_speed_change(self, val):
        speed = float(val)
        tag = " (慢動作)" if speed < 1.0 else (" (原速)" if speed == 1.0 else " (倍速)")
        self.lbl_speed_val.config(text=f"{speed:.1f}x{tag}")
        self.update_skip_prediction()

    def update_skip_prediction(self):
        speed = self.speed_slider.get()
        if not self.var_enable_skipping.get():
            self.lbl_prediction.config(text="[非跳幀模式：逐格 1 幀解算 (100% 採樣)]", foreground="blue")
        else:
            if speed < 1.0:
                self.lbl_prediction.config(text="[慢動作：每 1 影格計算一次 (延時播放)]", foreground="teal")
            else:
                skip = int(speed)
                sampling_pct = (1.0 / skip) * 100
                self.lbl_prediction.config(
                    text=f"[跳幀預測：每 {skip} 影格計算 1 次 (採樣率約 {sampling_pct:.0f}%)]", 
                    foreground="crimson" if skip >= 3 else "darkgreen"
                )

    def preprocess_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        enhanced_gray = cv2.equalizeHist(gray)
        return cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

    def open_video(self):
        if self.cap is not None:
            self.cap.release()
            
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")])
        if not self.video_path:
            return
        
        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0: self.fps = 30.0
        
        ret, self.current_frame = self.cap.read()
        if ret:
            self.first_frame = self.current_frame.copy()
            self.frame_data.clear()
            self.origin_x = None
            self.origin_y = None
            self.pixel_per_unit = None
            self.unit_name = "px"
            self.tracking = False
            self.frame_count = 0
            self.lbl_scale_info.config(text="比例尺：未設定 (以像素 px 為基準)")
            
            self.display_frame(self.current_frame)
            self.lbl_status.config(text=f"狀態：影片已載入 (FPS: {self.fps:.2f})")
            self.btn_scale.config(state=tk.NORMAL)
            self.btn_select.config(state=tk.NORMAL)
            self.btn_start.config(state=tk.DISABLED)
            self.btn_fast_start.config(state=tk.DISABLED)
            self.btn_export.config(state=tk.DISABLED)
            self.update_all_charts()
            self.update_skip_prediction()

    def display_frame(self, frame):
        frame_resized = cv2.resize(frame, (640, 480))
        cv2_im = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2_im)
        self.photo = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def start_calibration(self):
        if self.cap is None: return
        self.calibrating = True
        self.scale_points = []
        self.lbl_status.config(text="比例尺模式：請在畫面上「點選兩點」標註已知長度的物體")
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        _, self.current_frame = self.cap.read()
        self.display_frame(self.current_frame)

    def on_mouse_press(self, event):
        if self.calibrating:
            orig_h, orig_w, _ = self.current_frame.shape
            real_x = int(event.x * (orig_w / 640))
            real_y = int(event.y * (orig_h / 480))
            
            self.scale_points.append((real_x, real_y))
            self.canvas.create_oval(event.x-4, event.y-4, event.x+4, event.y+4, fill="yellow", tags="scale_dot")
            
            if len(self.scale_points) == 2:
                self.calibrating = False
                p1, p2 = self.scale_points[0], self.scale_points[1]
                pixel_dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                
                real_length = simpledialog.askfloat("比例尺設定", "請輸入這兩點之間的實際物理長度：", minvalue=0.001)
                if real_length:
                    unit = simpledialog.askstring("單位設定", "請輸入物理單位 (例如 mm, cm, m):", initialvalue="mm")
                    self.unit_name = unit if unit else "mm"
                    self.pixel_per_unit = pixel_dist / real_length
                    
                    self.lbl_scale_info.config(text=f"比例尺：1 {self.unit_name} = {self.pixel_per_unit:.2f} 像素 (px)")
                    self.lbl_status.config(text="比例尺設定成功！請點選 '2. 框選目標'")
                    self.update_all_charts()
                else:
                    self.lbl_status.config(text="已取消比例尺設定。")
                self.canvas.delete("scale_dot")
            return

        if not self.selecting: return
        self.init_x, self.init_y = event.x, event.y
        if self.rect: self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(self.init_x, self.init_y, self.init_x, self.init_y, outline="red", width=2)

    def on_mouse_drag(self, event):
        if self.calibrating or not self.selecting: return
        self.cur_x, self.cur_y = event.x, event.y
        self.canvas.coords(self.rect, self.init_x, self.init_y, self.cur_x, self.cur_y)

    def on_mouse_release(self, event):
        if self.calibrating or not self.selecting: return
        self.selecting = False
        
        x = min(self.init_x, self.cur_x)
        y = min(self.init_y, self.cur_y)
        w = abs(self.init_x - self.cur_x)
        h = abs(self.init_y - self.cur_y)
        
        if w < 5 or h < 5: return

        orig_h, orig_w, _ = self.current_frame.shape
        self.bbox = (int(x * (orig_w / 640)), int(y * (orig_h / 480)), int(w * (orig_w / 640)), int(h * (orig_h / 480)))
        
        self.init_clean_tracker()
        self.lbl_status.config(text="目標已鎖定！可選擇『播放追蹤』或『⚡快速出結果』")
        self.btn_start.config(state=tk.NORMAL)
        self.btn_fast_start.config(state=tk.NORMAL)

    def init_clean_tracker(self):
        try:
            self.tracker = cv2.legacy.TrackerCSRT_create()
        except AttributeError:
            self.tracker = cv2.TrackerCSRT_create()
            
        base_frame = self.first_frame if self.first_frame is not None else self.current_frame
        self.tracker.init(self.preprocess_frame(base_frame), self.bbox)

    def start_selection(self):
        if self.cap is None: return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        _, self.current_frame = self.cap.read()
        self.first_frame = self.current_frame.copy()
        self.display_frame(self.current_frame)
        self.frame_data.clear()
        self.origin_x = None
        self.origin_y = None
        self.frame_count = 0
        self.lost_frames_counter = 0
        self.selecting = True
        self.lbl_status.config(text="狀態：請在畫面上框選追蹤目標物")
        self.update_all_charts()

    # ==========================================
    # 嚴謹物理運動學解算核心 (Kinematics)
    # ==========================================
    def process_kinematics(self, cx, cy, frame_idx):
        factor = self.pixel_per_unit if self.pixel_per_unit else 1.0
        phys_x = cx / factor
        phys_y = cy / factor
        
        curr_time = frame_idx / self.fps
        
        if self.origin_x is None:
            self.origin_x = phys_x
            self.origin_y = phys_y
            
        disp_x = phys_x - self.origin_x
        disp_y = phys_y - self.origin_y
        disp_total = np.sqrt(disp_x**2 + disp_y**2)
        
        vel_x, vel_y, speed = 0.0, 0.0, 0.0
        accel_x, accel_y, accel_mag = 0.0, 0.0, 0.0
        jerk_mag = 0.0
        tot_path = 0.0
        
        if len(self.frame_data) > 0:
            prev = self.frame_data[-1]
            dt = curr_time - prev["Time"]
            if dt <= 0: dt = 1.0 / self.fps
            
            dx = phys_x - prev["Coord_X"]
            dy = phys_y - prev["Coord_Y"]
            
            step_px = np.sqrt((cx - prev["Raw_X"])**2 + (cy - prev["Raw_Y"])**2)
            
            # 濾除 <= 5px 抖動雜訊
            if step_px > 5.0:
                tot_path = prev.get("Path_Length", 0.0) + np.sqrt(dx**2 + dy**2)
            else:
                tot_path = prev.get("Path_Length", 0.0)
            
            # 向量速度與標量速率
            vel_x = dx / dt
            vel_y = dy / dt
            speed = np.sqrt(vel_x**2 + vel_y**2)
            
            # 向量加速度與大小
            accel_x = (vel_x - prev["Vel_X"]) / dt
            accel_y = (vel_y - prev["Vel_Y"]) / dt
            accel_mag = np.sqrt(accel_x**2 + accel_y**2)
            
            # 向量急動度與大小
            jerk_x = (accel_x - prev["Accel_X"]) / dt
            jerk_y = (accel_y - prev["Accel_Y"]) / dt
            jerk_mag = np.sqrt(jerk_x**2 + jerk_y**2)
            
        return {
            "Frame": frame_idx, "Time": curr_time,
            "Raw_X": cx, "Raw_Y": cy,
            "Coord_X": phys_x, "Coord_Y": phys_y,
            "Disp_X": disp_x, "Disp_Y": disp_y,
            "Disp_Total": disp_total,
            "Path_Length": tot_path,
            "Vel_X": vel_x, "Vel_Y": vel_y,
            "Speed": speed,
            "Accel_X": accel_x, "Accel_Y": accel_y,
            "Accel_Mag": accel_mag,
            "Jerk_Mag": jerk_mag
        }

    def prepare_for_reanalysis(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        _, self.current_frame = self.cap.read()
        self.frame_data.clear()
        self.origin_x = None
        self.origin_y = None
        self.frame_count = 0
        self.lost_frames_counter = 0
        self.init_clean_tracker()
        self.update_all_charts()

    def start_tracking(self):
        if self.bbox is None: return
        if len(self.frame_data) > 0 or self.frame_count > 0:
            self.prepare_for_reanalysis()
            
        self.tracking = True
        self.set_buttons_state(tk.DISABLED)
        self.update_loop()

    def start_fast_tracking(self):
        if self.bbox is None: return
        if len(self.frame_data) > 0 or self.frame_count > 0:
            self.prepare_for_reanalysis()
            
        self.set_buttons_state(tk.DISABLED)
        self.lbl_status.config(text="⚡ 正在以極速計算全量科學數據...")
        self.window.update()
        
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            
            self.frame_count += 1
            success, bbox = self.tracker.update(self.preprocess_frame(frame))
            
            if success:
                self.lost_frames_counter = 0
                x, y, w, h = [int(v) for v in bbox]
                row = self.process_kinematics(int(x + w/2), int(y + h/2), self.frame_count)
                self.frame_data.append(row)
            else:
                self.lost_frames_counter += 1
                if self.lost_frames_counter >= self.lost_frames_limit:
                    break
        
        self.update_all_charts()
        self.end_tracking("⚡ 快速解算完成！可再次點擊重新分析。")

    def update_loop(self):
        if not self.tracking: return
        
        speed_factor = self.speed_slider.get()
        if speed_factor <= 0: speed_factor = 1.0
        
        skip_frames = int(speed_factor) if (self.var_enable_skipping.get() and speed_factor >= 1.0) else 1
            
        ret, frame = False, None
        for _ in range(skip_frames):
            ret, frame = self.cap.read()
            if not ret: break
            self.frame_count += 1
            
        if not ret or frame is None:
            self.update_all_charts()
            self.end_tracking("影片播放結束。可隨時再次點擊按鈕重新分析！")
            return
            
        self.current_frame = frame.copy()
        success, bbox = self.tracker.update(self.preprocess_frame(frame))
        
        if success:
            self.lost_frames_counter = 0
            x, y, w, h = [int(v) for v in bbox]
            cx, cy = int(x + w/2), int(y + h/2)
            
            row = self.process_kinematics(cx, cy, self.frame_count)
            self.frame_data.append(row)
            
            if len(self.frame_data) > 1:
                for i in range(max(1, len(self.frame_data)-40), len(self.frame_data)):
                    pt1 = (self.frame_data[i-1]["Raw_X"], self.frame_data[i-1]["Raw_Y"])
                    pt2 = (self.frame_data[i]["Raw_X"], self.frame_data[i]["Raw_Y"])
                    cv2.line(frame, pt1, pt2, (0, 165, 255), 2, cv2.LINE_AA)
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.circle(frame, (cx, cy), 6, (255, 0, 0), -1)
            cv2.putText(frame, f"v: {row['Speed']:.1f} {self.unit_name}/s | |a|: {row['Accel_Mag']:.1f} {self.unit_name}/s2", 
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
            
            if len(self.frame_data) % 5 == 0:
                self.update_all_charts()
        else:
            self.lost_frames_counter += 1
            if self.lost_frames_counter >= self.lost_frames_limit:
                self.update_all_charts()
                self.end_tracking("追蹤遺失超時。可再次點擊按鈕重新分析！")
                return
            
        self.display_frame(frame)
        
        delay_ms = max(1, int((1000.0 / self.fps) / speed_factor)) if (not self.var_enable_skipping.get() or speed_factor < 1.0) else 1
        self.window.after(delay_ms, self.update_loop)

    # ==========================================
    # 獨立卡片圖表架構 (每張圖專屬 Canvas，杜絕重疊覆蓋)
    # ==========================================
    def rebuild_chart_container(self):
        for widget in self.scrollable_inner_frame.winfo_children():
            widget.destroy()
        self.chart_cards.clear()
        
        selected_metrics = [k for k, v in self.metric_vars.items() if v.get()]
        if not selected_metrics:
            lbl = ttk.Label(self.scrollable_inner_frame, text="請在上方勾選至少一項物理量以顯示圖表", font=("Microsoft JhengHei", 10), foreground="gray")
            lbl.pack(pady=40)
            return
            
        for key in selected_metrics:
            label_name, unit_pattern, color = self.metric_defs[key]
            
            card_frame = ttk.Frame(self.scrollable_inner_frame, relief="ridge", borderwidth=1, padding=4)
            card_frame.pack(fill=tk.X, expand=True, padx=4, pady=4)
            
            fig, ax = plt.subplots(figsize=(5.6, 1.8), facecolor='#ffffff')
            fig.subplots_adjust(left=0.20, right=0.96, top=0.88, bottom=0.26)
            
            canvas = FigureCanvasTkAgg(fig, master=card_frame)
            canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            btn_view = ttk.Button(
                card_frame, text=f"📊 查看\n{label_name.split()[0]} 數據", 
                style="White.TButton",
                command=lambda k=key: self.open_single_metric_viewer(k)
            )
            btn_view.pack(side=tk.RIGHT, padx=8, pady=10)
            
            self.chart_cards[key] = {
                "fig": fig,
                "ax": ax,
                "canvas": canvas,
                "color": color,
                "label": label_name,
                "unit": unit_pattern
            }
            
        self.update_all_charts()

    def update_all_charts(self):
        df = pd.DataFrame(self.frame_data) if self.frame_data else pd.DataFrame()
        smooth_active = self.var_smooth_chart.get()
        
        for key, item in self.chart_cards.items():
            ax = item["ax"]
            ax.cla()
            ax.grid(True, linestyle="--", alpha=0.5)
            
            actual_unit = item["unit"].format(unit=self.unit_name)
            ax.set_ylabel(f"{item['label']}\n({actual_unit})", fontproperties='Microsoft JhengHei', fontsize=8, labelpad=8)
            ax.set_xlabel("時間 Time (s)", fontproperties='Microsoft JhengHei', fontsize=8)
            ax.tick_params(axis='both', labelsize=8)
            
            if not df.empty and key in df.columns:
                t = df["Time"].values
                y = df[key].values
                
                if smooth_active and len(t) >= 4:
                    try:
                        t_dense = np.linspace(t.min(), t.max(), len(t) * 5)
                        spl = make_interp_spline(t, y, k=3)
                        ax.plot(t_dense, spl(t_dense), color=item["color"], linewidth=1.6)
                    except Exception:
                        ax.plot(t, y, color=item["color"], linewidth=1.5)
                else:
                    ax.plot(t, y, color=item["color"], linewidth=1.5)
                    
            item["canvas"].draw()

    # 單一數據即時表格閱覽器
    def open_single_metric_viewer(self, metric_key):
        if not self.frame_data:
            messagebox.showwarning("提示", "目前尚無任何追蹤數據！")
            return
            
        df = pd.DataFrame(self.frame_data)
        label_name, unit_pattern, _ = self.metric_defs[metric_key]
        actual_unit = unit_pattern.format(unit=self.unit_name)
        
        sub_win = tk.Toplevel(self.window)
        sub_win.title(f"數據快速閱覽 - {label_name}")
        sub_win.geometry("520x450")
        
        top_bar = ttk.Frame(sub_win, padding=6)
        top_bar.pack(fill=tk.X)
        
        val_series = df[metric_key]
        summary_text = f"總取樣點：{len(df)} | 最大值：{val_series.max():.2f} {actual_unit} | 平均值：{val_series.mean():.2f} {actual_unit}"
        ttk.Label(top_bar, text=summary_text, font=("Microsoft JhengHei", 9, "bold")).pack(side=tk.LEFT)
        
        # 表格元件
        tree_frame = ttk.Frame(sub_win)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
        
        tree = ttk.Treeview(tree_frame, columns=("Frame", "Time", "Value"), show="headings", height=15)
        tree.heading("Frame", text="影格 (Frame)")
        tree.heading("Time", text="時間 (s)")
        tree.heading("Value", text=f"{label_name} ({actual_unit})")
        
        tree.column("Frame", width=100, anchor="center")
        tree.column("Time", width=120, anchor="center")
        tree.column("Value", width=200, anchor="center")
        
        tree_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=tree_scroll.set)
        
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        for idx, row in df.iterrows():
            tree.insert("", "end", values=(int(row["Frame"]), f"{row['Time']:.3f}", f"{row[metric_key]:.4f}"))

    # 自選輸出欄位的導出 CSV 視窗
    def open_export_dialog(self):
        if not self.frame_data:
            messagebox.showwarning("警告", "目前尚無任何追蹤數據可匯出！")
            return
            
        export_win = tk.Toplevel(self.window)
        export_win.title("導出 CSV - 勾選匯出欄位")
        export_win.geometry("450x480")
        export_win.resizable(False, False)
        
        ttk.Label(export_win, text="請勾選要匯出至 CSV 的科學計量：", font=("Microsoft JhengHei", 10, "bold")).pack(pady=10)
        
        check_frame = ttk.Frame(export_win, padding=10)
        check_frame.pack(fill=tk.BOTH, expand=True)
        
        export_vars = {}
        for key, (label_text, _, _) in self.metric_defs.items():
            initial_val = self.metric_vars.get(key, tk.BooleanVar(value=True)).get()
            var = tk.BooleanVar(value=initial_val)
            export_vars[key] = var
            chk = ttk.Checkbutton(check_frame, text=label_text, variable=var)
            chk.pack(anchor="w", pady=2)
            
        btn_action_frame = ttk.Frame(export_win, padding=10)
        btn_action_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        def select_all(state):
            for v in export_vars.values(): v.set(state)
            
        ttk.Button(btn_action_frame, text="全選", command=lambda: select_all(True), style="White.TButton").pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_action_frame, text="全不選", command=lambda: select_all(False), style="White.TButton").pack(side=tk.LEFT, padx=4)
        
        def execute_export():
            chosen_metrics = [k for k, v in export_vars.items() if v.get()]
            if not chosen_metrics:
                messagebox.showwarning("提示", "請至少勾選一個匯出欄位！")
                return
                
            save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
            if not save_path: return
            
            df = pd.DataFrame(self.frame_data)
            export_cols = ["Frame", "Time"] + chosen_metrics
            df[export_cols].to_csv(save_path, index=False)
            
            export_win.destroy()
            messagebox.showinfo("成功", f"數據已儲存！共匯出 {len(chosen_metrics)} 項物理量至：\n{save_path}")
            
        ttk.Button(btn_action_frame, text="確認導出 CSV", command=execute_export, style="White.TButton").pack(side=tk.RIGHT, padx=4)

    def set_buttons_state(self, state):
        self.btn_open.config(state=state)
        self.btn_scale.config(state=state)
        self.btn_select.config(state=state)
        self.btn_start.config(state=state)
        self.btn_fast_start.config(state=state)

    def end_tracking(self, message):
        self.tracking = False
        self.set_buttons_state(tk.NORMAL)
        self.btn_export.config(state=tk.NORMAL)
        self.lbl_status.config(text=f"狀態：{message}")

if __name__ == "__main__":
    root = tk.Tk()
    app = UniversalPhysicsTrackerApp(root)
    root.mainloop()
