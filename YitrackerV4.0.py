import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import cv2
from PIL import Image, ImageTk
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.interpolate import make_interp_spline
import platform
import os
import math

# ==========================================
# 1. 國際化多國語言字典 (i18n)
# ==========================================
TRANSLATIONS = {
    "zh_TW": {
        "title": "科研級運動追蹤分析工具 V4.0 Pro (通用流體/剛體模式版)",
        "open_video": "📁 開啟影片 (建立新專案)",
        "set_scale": "📐 1. 設定比例尺",
        "select_target": "🎯 2. 手動框選",
        "color_target": "🎨 2. 色彩譜與方位鎖定",
        "track_play": "▶ 3. 播放追蹤",
        "fast_track": "⚡ 3. 快速解算",
        "export_center": "💾 4. 匯出成果",
        "guide_title": "📖 操作流程指南 (Workflow Guide)",
        "guide_steps": "步驟：[1. 開啟影片] ➔ [2. 標註比例尺] ➔ [3. 剪輯 A/B 點] ➔ [4. 選擇模式並鎖定目標] ➔ [5. 追蹤解算並匯出]",
        "tracking_mode_label": "🎯 核心追蹤模式 (依實驗目標選擇)：",
        "algo_settings": "⚙️ 影像預處理與輔助設定",
        "enhance_label": "影像增強：",
        "kalman_enable": "啟用卡爾曼濾波 (剛體抗遮擋與平滑)",
        "trim_panel": "✂️ 影片剪輯與分析區間",
        "set_start_frame": "設為起點 [A]",
        "set_end_frame": "設為終點 [B]",
        "play_sampling": "⏩ 播放與取樣控制",
        "reset_speed": "重設 1.0x",
        "skip_frames": "啟用跳幀加速 (倍速時生效)",
        "predict_single": "[逐格解算模式 (100% 採樣)]",
        "predict_slow": "[慢動作模式 (延時解算)]",
        "predict_skip": "[跳幀解算：每 {skip} 幀計算一次 (約 {pct:.0f}% 採樣)]",
        "scale_not_set": "比例尺：未設定 (px)",
        "scale_set": "比例尺：1 {unit} = {scale:.2f} px",
        "smooth_chart": "🌊 啟用圖表曲線平滑化 (Spline)",
        "metrics_panel": "🔬 勾選顯示的物理量 (即時繪圖 / 獨立檢視)",
        "view_data": "📊 查看數據",
        "info_btn": "ℹ️ 關於",
        "status_ready": "狀態：專案準備中，請先開啟影片建立專案",
        "status_loaded": "狀態：新專案已初始化 (FPS: {fps:.2f}, 總影格: {total})",
        "status_calibrating": "比例尺模式：請在畫面上「點選兩點」已知長度",
        "status_color_pick": "取色模式：請在預覽畫面上「點選」目標物顏色",
        "status_locked": "目標已鎖定！可點選播放追蹤或快速解算",
        "status_tracking_done": "區間分析完成！",
        "status_lost": "目標遺失或消失，追蹤結束。",
        "status_fast_running": "⚡ 正在解算剪輯區間內的全量科學數據...",
        "select_target_prompt": "請在畫面拖曳滑鼠框選目標物",
        "scale_len_title": "比例尺長度",
        "scale_len_prompt": "請輸入這兩點之間的物理長度：",
        "scale_unit_title": "單位設定",
        "scale_unit_prompt": "請輸入物理長度單位 (例: mm, cm, m):",
        "no_data_warn": "目前尚無任何追蹤數據！",
        "reset_confirm": "偵測到已有進行中的分析數據，開啟新影片將重設所有專案設定與軌跡，是否繼續？",
        "time_axis": "時間 Time (s)",
        "frame_col": "影格 (Frame)",
        "time_col": "時間 (Time s)",
        "summary_fmt": "樣本數：{n} | 最大值：{max_v:.2f} {unit} | 平均值：{mean_v:.2f} {unit}",
        "prep_options": ["CLAHE (高對比自適應)", "灰階去噪 (高斯濾波)", "原始色彩 (無處理)"],
        "export_modal_title": "科學成果匯出中心 (CSV / 圖表)",
        "export_step1": "1. 勾選要匯出的物理量：",
        "export_step2": "2. 輸出設定 (圖表與格式)：",
        "export_combine_label": "將所有數據圖表整合排列成一張 (科研報表排版)",
        "export_format_label": "圖表格式：",
        "export_include_csv": "同時輸出 CSV 原始數據檔",
        "btn_export_action": "選擇路徑並開始匯出",
        "select_all": "全選",
        "deselect_all": "全不選",
        "export_success": "匯出完成！檔案已儲存至：\n{path}",
        "dir_options": ["最靠下方 (羽流最底前端)", "最靠近畫面中心", "最靠上方 (頂部)", "最靠左方", "最靠右方"],
        "color_modal_title": "彩譜色盤與方位目標鎖定",
        "color_tol_label": "色彩容許度 (Tolerance):",
        "dir_select_label": "目標優先鎖定方位：",
        "palette_label": "🎨 快速彩譜預設色 (點擊載入標準色相)：",
        "metrics": {
            "Coord_X": ("X 軸座標 (X)", "{unit}"),
            "Coord_Y": ("Y 軸座標 (Y)", "{unit}"),
            "Disp_X": ("X 軸位移 (ΔX)", "{unit}"),
            "Disp_Y": ("Y 軸位移 (ΔY)", "{unit}"),
            "Disp_Total": ("總位移量 (|Δr|)", "{unit}"),
            "Path_Length": ("累計路徑長", "{unit}"),
            "Vel_X": ("X 軸速度分量 (Vx)", "{unit}/s"),
            "Vel_Y": ("Y 軸速度分量 (Vy)", "{unit}/s"),
            "Speed": ("瞬時速率 (|v|)", "{unit}/s"),
            "Accel_X": ("X 軸加速度 (ax)", "{unit}/s²"),
            "Accel_Y": ("Y 軸加速度 (ay)", "{unit}/s²"),
            "Accel_Mag": ("瞬時加速度大小 (|a|)", "{unit}/s²"),
            "Jerk_Mag": ("瞬時急動度 (|J|)", "{unit}/s³")
        }
    },
    "en": {
        "title": "Scientific Motion Physics Tracker V4.0 Pro",
        "open_video": "📁 Open Video (New Project)",
        "set_scale": "📐 1. Set Scale",
        "select_target": "🎯 2. Select ROI",
        "color_target": "🎨 2. Palette & Bearing",
        "track_play": "▶ 3. Track Range",
        "fast_track": "⚡ 3. Fast Solve",
        "export_center": "💾 4. Export Results",
        "guide_title": "📖 Workflow Guide",
        "guide_steps": "Steps: [1. Open Video] ➔ [2. Set Scale] ➔ [3. Mark A/B Range] ➔ [4. Select Mode & Target] ➔ [5. Track & Export]",
        "tracking_mode_label": "🎯 Tracking Mode & Suitable Target:",
        "algo_settings": "⚙ Preprocessing & Auxiliary Settings",
        "enhance_label": "Enhance:",
        "kalman_enable": "Enable Kalman Filter (Occlusion & Smooth)",
        "trim_panel": "✂️ Clip Trimming & Analysis Range",
        "set_start_frame": "Mark Start [A]",
        "set_end_frame": "Mark End [B]",
        "play_sampling": "⏩ Playback & Sampling Rate",
        "reset_speed": "Reset 1.0x",
        "skip_frames": "Enable Frame Skipping (Speed > 1x)",
        "predict_single": "[Continuous Analysis (100% Sampling)]",
        "predict_slow": "[Slow Motion (Delay Playback)]",
        "predict_skip": "[Skipping: 1 calculation per {skip} frames (~{pct:.0f}% sample)]",
        "scale_not_set": "Scale: Uncalibrated (px)",
        "scale_set": "Scale: 1 {unit} = {scale:.2f} px",
        "smooth_chart": "🌊 Enable Curve Smoothing (Cubic Spline)",
        "metrics_panel": "🔬 Monitored Kinematics (Real-time Plots)",
        "view_data": "📊 View Data",
        "info_btn": "ℹ️ About",
        "status_ready": "Status: Project ready. Open video to initialize project.",
        "status_loaded": "Status: Project initialized (FPS: {fps:.2f}, Total: {total})",
        "status_calibrating": "Calibration: Click 2 points with known physical distance.",
        "status_color_pick": "Color Picker: Click on target in video frame.",
        "status_locked": "Target locked. Ready for Range Track or Fast Solve.",
        "status_tracking_done": "Range analysis completed.",
        "status_lost": "Target lost (timeout exceeded).",
        "status_fast_running": "⚡ Computing full kinematic parameters within clip range...",
        "select_target_prompt": "Drag mouse over the canvas to select the target.",
        "scale_len_title": "Scale Distance",
        "scale_len_prompt": "Enter the real physical distance between points:",
        "scale_unit_title": "Unit Settings",
        "scale_unit_prompt": "Enter unit name (e.g., mm, cm, m):",
        "no_data_warn": "No tracking data available yet!",
        "reset_confirm": "Active tracking session found. Loading a new video will reset all project data and buttons. Proceed?",
        "time_axis": "Time (s)",
        "frame_col": "Frame #",
        "time_col": "Time (s)",
        "summary_fmt": "Samples: {n} | Max: {max_v:.2f} {unit} | Mean: {mean_v:.2f} {unit}",
        "prep_options": ["CLAHE (Adaptive)", "Gaussian Denoise", "Raw Color"],
        "export_modal_title": "Export Center (CSV & Visual Figures)",
        "export_step1": "1. Select Kinematics to Export:",
        "export_step2": "2. Figure & File Configuration:",
        "export_combine_label": "Combine all plots into a single scientific figure layout",
        "export_format_label": "Figure Format:",
        "export_include_csv": "Include Raw Data Table (.csv)",
        "btn_export_action": "Choose Destination & Save",
        "select_all": "Select All",
        "deselect_all": "Deselect All",
        "export_success": "Export successfully saved to:\n{path}",
        "dir_options": ["Closest to Bottom (Plume Front)", "Closest to Center", "Closest to Top", "Closest to Left", "Closest to Right"],
        "color_modal_title": "Palette & Directional Target Settings",
        "color_tol_label": "Color Tolerance:",
        "dir_select_label": "Target Prioritized Direction:",
        "palette_label": "🎨 Predefined Color Palette (Click to load HSV):",
        "metrics": {
            "Coord_X": ("X Coordinate (X)", "{unit}"),
            "Coord_Y": ("Y Coordinate (Y)", "{unit}"),
            "Disp_X": ("Displacement X (ΔX)", "{unit}"),
            "Disp_Y": ("Displacement Y (ΔY)", "{unit}"),
            "Disp_Total": ("Net Displacement (|Δr|)", "{unit}"),
            "Path_Length": ("Traveled Path Length", "{unit}"),
            "Vel_X": ("Velocity X (Vx)", "{unit}/s"),
            "Vel_Y": ("Velocity Y (Vy)", "{unit}/s"),
            "Speed": ("Instantaneous Speed (|v|)", "{unit}/s"),
            "Accel_X": ("Acceleration X (ax)", "{unit}/s²"),
            "Accel_Y": ("Acceleration Y (ay)", "{unit}/s²"),
            "Accel_Mag": ("Total Acceleration (|a|)", "{unit}/s²"),
            "Jerk_Mag": ("Instantaneous Jerk (|J|)", "{unit}/s³")
        }
    }
}

def apply_adaptive_matplotlib_font(lang_code="zh_TW"):
    sys_name = platform.system()
    if lang_code in ["zh_TW", "zh_CN"]:
        fonts = ['Microsoft JhengHei', 'SimHei', 'PingFang TC', 'DejaVu Sans'] if sys_name == "Windows" else ['PingFang TC', 'Arial Unicode MS', 'DejaVu Sans']
    else:
        fonts = ['Segoe UI', 'Helvetica Neue', 'Arial', 'DejaVu Sans']
    plt.rcParams['font.sans-serif'] = fonts
    plt.rcParams['axes.unicode_minus'] = False

class UniversalKalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.04
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.4
        self.is_initialized = False

    def init(self, cx, cy):
        self.kf.statePre = np.array([[float(cx)], [float(cy)], [0.0], [0.0]], np.float32)
        self.kf.statePost = np.array([[float(cx)], [float(cy)], [0.0], [0.0]], np.float32)
        self.is_initialized = True

    def predict(self):
        if not self.is_initialized: return 0.0, 0.0
        pred = self.kf.predict()
        return float(pred[0, 0]), float(pred[1, 0])

    def correct(self, cx, cy):
        measurement = np.array([[float(cx)], [float(cy)]], np.float32)
        est = self.kf.correct(measurement)
        return float(est[0, 0]), float(est[1, 0])

class UniversalPhysicsTrackerApp:
    def __init__(self, window):
        self.window = window
        self.window.geometry("1600x1020")
        self.window.minsize(1280, 840)
        self.current_lang = "zh_TW"
        
        apply_adaptive_matplotlib_font(self.current_lang)
        self.window.title(self.t("title"))
        
        self.colors = {
            "bg_main": "#1e222d", "bg_card": "#262b36", "bg_card_border": "#363c4d",
            "text_primary": "#ffffff", "text_secondary": "#a0a8b7",
            "accent": "#00adb5", "accent_hover": "#00cbd4", "canvas_bg": "#12151b"
        }
        self.window.configure(bg=self.colors["bg_main"])
        
        # 專案資料狀態
        self.video_path = None
        self.cap = None
        self.tracker = None
        self.tracking = False
        self.selecting = False
        self.color_picking = False
        self.bbox = None
        self.first_frame = None
        self.current_frame = None
        self.frame_data = []
        self.fps = 30.0
        self.total_frames = 0
        self.current_frame_idx = 0
        
        self.trim_start = 0
        self.trim_end = 0
        
        # 追蹤模式定義 (名稱 + 適用目標說明)
        self.mode_options = [
            "🌊 流體/羽流前端最底端追蹤 (適用：擴散墨水、下沉羽流、沉澱反應、熔滴)",
            "🎯 CSRT 剛體質心高精追蹤 (適用：小球、單擺、車輛、剛體變形小目標)",
            "⚡ KCF 高速剛體追蹤 (適用：高速運動物體、碰撞彈跳、無形變目標)",
            "🎈 色彩輪廓質心動態追蹤 (適用：單純色塊、標記貼紙、均勻變形色塊)"
        ]
        self.var_tracking_mode = tk.StringVar(value=self.mode_options[0])
        
        self.target_hsv = (60, 220, 220) # 預設綠色
        self.color_tolerance = 25
        self.target_direction = "最靠下方 (羽流最底前端)"
        
        # 預設常見彩譜定義
        self.preset_palette = {
            "亮綠色 (染料)": (60, 220, 220),
            "湛藍色 (墨水)": (115, 220, 220),
            "鮮紅色 (指示劑)": (0, 220, 220),
            "鮮黃色 (沉澱物)": (30, 220, 220),
            "亮橘色 (油滴)": (15, 230, 240),
            "紫羅蘭 (高錳酸鉀)": (140, 200, 220),
            "純白 (懸浮反光)": (0, 10, 245),
            "黑色 (碳黑/磁粉)": (0, 0, 30)
        }
        
        self.origin_x = None
        self.origin_y = None
        self.lost_frames_limit = 40
        self.lost_frames_counter = 0
        
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.kalman = UniversalKalmanFilter()
        
        self.calibrating = False
        self.scale_points = []
        self.pixel_per_unit = None  
        self.unit_name = "px"       
        
        self.init_x, self.init_y = 0, 0
        self.cur_x, self.cur_y = 0, 0
        self.rect = None
        
        self.var_enable_skipping = tk.BooleanVar(value=True)
        self.var_smooth_chart = tk.BooleanVar(value=True)
        self.var_preprocess_mode = tk.StringVar()
        self.var_use_kalman = tk.BooleanVar(value=True)
        self.var_lang_choice = tk.StringVar(value="繁體中文")
        
        self.metric_colors = {
            "Coord_X": "#2563eb", "Coord_Y": "#1e3a8a",
            "Disp_X": "#0284c7", "Disp_Y": "#0369a1", "Disp_Total": "#0f766e",
            "Path_Length": "#b45309",
            "Vel_X": "#16a34a", "Vel_Y": "#15803d", "Speed": "#7c3aed",
            "Accel_X": "#dc2626", "Accel_Y": "#b91c1c", "Accel_Mag": "#e11d48",
            "Jerk_Mag": "#ea580c"
        }
        
        self.metric_vars = {}
        self.metric_checkboxes = {}
        self.chart_cards = {}
        
        self.setup_styles()
        self.setup_ui()
        self.change_language("zh_TW", force=True)

    def t(self, key):
        return TRANSLATIONS.get(self.current_lang, {}).get(key, key)

    def setup_styles(self):
        self.style = ttk.Style()
        try: self.style.theme_use("clam")
        except Exception: pass
            
        self.style.configure(".", background=self.colors["bg_main"], foreground=self.colors["text_primary"])
        self.style.configure("TFrame", background=self.colors["bg_main"])
        self.style.configure("Card.TFrame", background=self.colors["bg_card"], relief="solid", borderwidth=1)
        self.style.configure("TLabel", background=self.colors["bg_main"], foreground=self.colors["text_primary"], font=("Segoe UI", 9))
        self.style.configure("TCheckbutton", background=self.colors["bg_card"], foreground=self.colors["text_primary"], font=("Segoe UI", 9))
        self.style.map("TCheckbutton", background=[('active', self.colors["bg_card"])])
        self.style.configure("TLabelframe", background=self.colors["bg_card"], bordercolor=self.colors["bg_card_border"], relief="solid")
        self.style.configure("TLabelframe.Label", background=self.colors["bg_card"], foreground=self.colors["accent"], font=("Segoe UI", 10, "bold"))

        self.style.configure("Modern.TButton", background="#3b4252", foreground="#ffffff", font=("Segoe UI", 9, "bold"), padding=(8, 4), relief="flat")
        self.style.map("Modern.TButton", background=[('pressed', '#2e3440'), ('active', '#4c566a'), ('disabled', '#2a2e39')], foreground=[('disabled', '#6b7280')])
        
        self.style.configure("Accent.TButton", background=self.colors["accent"], foreground="#ffffff", font=("Segoe UI", 9, "bold"), padding=(8, 4), relief="flat")
        self.style.map("Accent.TButton", background=[('pressed', '#008b91'), ('active', self.colors["accent_hover"]), ('disabled', '#2a2e39')], foreground=[('disabled', '#6b7280')])

    def setup_ui(self):
        # 頂部導航列
        top_bar = tk.Frame(self.window, bg=self.colors["bg_card"], height=42)
        top_bar.pack(side=tk.TOP, fill=tk.X)
        
        self.lbl_app_title = tk.Label(top_bar, text=self.t("title"), bg=self.colors["bg_card"], fg="#00f7ff", font=("Segoe UI", 12, "bold"))
        self.lbl_app_title.pack(side=tk.LEFT, padx=14, pady=6)
        
        lang_box = tk.Frame(top_bar, bg=self.colors["bg_card"])
        lang_box.pack(side=tk.RIGHT, padx=14)
        tk.Label(lang_box, text="🌐 Language:", bg=self.colors["bg_card"], fg=self.colors["text_secondary"], font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=4)
        
        self.combo_lang = ttk.Combobox(lang_box, textvariable=self.var_lang_choice, values=["繁體中文", "English"], state="readonly", width=10)
        self.combo_lang.pack(side=tk.LEFT)
        self.combo_lang.bind("<<ComboboxSelected>>", lambda e: self.change_language("zh_TW" if self.var_lang_choice.get() == "繁體中文" else "en"))
        
        # 操作指引橫條
        self.guide_bar = tk.Frame(self.window, bg="#111827", padx=12, pady=6)
        self.guide_bar.pack(side=tk.TOP, fill=tk.X)
        
        self.lbl_guide_title = tk.Label(self.guide_bar, text=self.t("guide_title"), bg="#111827", fg="#38bdf8", font=("Segoe UI", 9, "bold"))
        self.lbl_guide_title.pack(side=tk.LEFT, padx=(4, 10))
        
        self.lbl_guide_steps = tk.Label(self.guide_bar, text=self.t("guide_steps"), bg="#111827", fg="#cbd5e1", font=("Segoe UI", 9))
        self.lbl_guide_steps.pack(side=tk.LEFT)

        # 主工作區
        main_workspace = tk.Frame(self.window, bg=self.colors["bg_main"])
        main_workspace.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        
        self.left_frame = tk.Frame(main_workspace, bg=self.colors["bg_main"], width=680)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 6), pady=4)
        
        # 步驟按鈕列
        self.btn_frame = tk.Frame(self.left_frame, bg=self.colors["bg_main"])
        self.btn_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))
        
        self.btn_open = ttk.Button(self.btn_frame, text=self.t("open_video"), command=self.open_video_project, style="Accent.TButton")
        self.btn_open.pack(side=tk.LEFT, padx=2)
        
        self.btn_scale = ttk.Button(self.btn_frame, text=self.t("set_scale"), command=self.start_calibration, state=tk.DISABLED, style="Modern.TButton")
        self.btn_scale.pack(side=tk.LEFT, padx=2)
        
        self.btn_select = ttk.Button(self.btn_frame, text=self.t("select_target"), command=self.start_selection, state=tk.DISABLED, style="Modern.TButton")
        self.btn_select.pack(side=tk.LEFT, padx=2)

        self.btn_color_pick = ttk.Button(self.btn_frame, text=self.t("color_target"), command=self.open_color_picker_settings, state=tk.DISABLED, style="Modern.TButton")
        self.btn_color_pick.pack(side=tk.LEFT, padx=2)
        
        self.btn_start = ttk.Button(self.btn_frame, text=self.t("track_play"), command=self.start_tracking, state=tk.DISABLED, style="Accent.TButton")
        self.btn_start.pack(side=tk.LEFT, padx=2)
        
        self.btn_fast_start = ttk.Button(self.btn_frame, text=self.t("fast_track"), command=self.start_fast_tracking, state=tk.DISABLED, style="Modern.TButton")
        self.btn_fast_start.pack(side=tk.LEFT, padx=2)
        
        self.btn_export = ttk.Button(self.btn_frame, text=self.t("export_center"), command=self.open_advanced_export_dialog, state=tk.DISABLED, style="Modern.TButton")
        self.btn_export.pack(side=tk.RIGHT, padx=2)

        # 核心追蹤模式選擇列 (整合至下拉選單並標註目標)
        self.frame_mode_select = ttk.LabelFrame(self.left_frame, text=self.t("tracking_mode_label"), padding=6)
        self.frame_mode_select.pack(side=tk.TOP, fill=tk.X, pady=3)
        
        self.combo_mode = ttk.Combobox(
            self.frame_mode_select, textvariable=self.var_tracking_mode, 
            values=self.mode_options, state="readonly", font=("Segoe UI", 9, "bold")
        )
        self.combo_mode.pack(fill=tk.X, padx=4, pady=2)

        # 影片剪輯與分析區間
        self.frame_trim = ttk.LabelFrame(self.left_frame, text=self.t("trim_panel"), padding=6)
        self.frame_trim.pack(side=tk.TOP, fill=tk.X, pady=3)
        
        trim_slider_row = tk.Frame(self.frame_trim, bg=self.colors["bg_card"])
        trim_slider_row.pack(fill=tk.X, pady=2)
        
        self.trim_slider = tk.Scale(
            trim_slider_row, from_=0, to=100, orient=tk.HORIZONTAL,
            highlightthickness=0, command=self.on_timeline_seek, 
            bg=self.colors["bg_card"], fg="#ffffff", troughcolor="#12151b", activebackground=self.colors["accent"]
        )
        self.trim_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        
        self.lbl_trim_info = tk.Label(self.frame_trim, text="區間：0 ~ 0 影格 (0.00s ~ 0.00s)", bg=self.colors["bg_card"], fg="#38bdf8", font=("Segoe UI", 8, "bold"))
        self.lbl_trim_info.pack(side=tk.LEFT, padx=6)
        
        self.btn_mark_a = ttk.Button(self.frame_trim, text=self.t("set_start_frame"), command=self.mark_trim_start, style="Modern.TButton", state=tk.DISABLED)
        self.btn_mark_a.pack(side=tk.RIGHT, padx=2)
        self.btn_mark_b = ttk.Button(self.frame_trim, text=self.t("set_end_frame"), command=self.mark_trim_end, style="Modern.TButton", state=tk.DISABLED)
        self.btn_mark_b.pack(side=tk.RIGHT, padx=2)

        # 影像預處理與輔助
        self.frame_algo = ttk.LabelFrame(self.left_frame, text=self.t("algo_settings"), padding=6)
        self.frame_algo.pack(side=tk.TOP, fill=tk.X, pady=3)
        
        algo_row = tk.Frame(self.frame_algo, bg=self.colors["bg_card"])
        algo_row.pack(fill=tk.X, pady=1)
        
        self.lbl_prep = tk.Label(algo_row, text=self.t("enhance_label"), bg=self.colors["bg_card"], fg=self.colors["text_primary"])
        self.lbl_prep.pack(side=tk.LEFT, padx=4)
        self.combo_prep = ttk.Combobox(algo_row, textvariable=self.var_preprocess_mode, values=self.t("prep_options"), width=20, state="readonly")
        self.combo_prep.current(0)
        self.combo_prep.pack(side=tk.LEFT, padx=4)
        
        self.chk_kalman = ttk.Checkbutton(algo_row, text=self.t("kalman_enable"), variable=self.var_use_kalman)
        self.chk_kalman.pack(side=tk.LEFT, padx=12)

        # 速度與取樣控制
        self.frame_speed = ttk.LabelFrame(self.left_frame, text=self.t("play_sampling"), padding=6)
        self.frame_speed.pack(side=tk.TOP, fill=tk.X, pady=3)
        
        slider_row = tk.Frame(self.frame_speed, bg=self.colors["bg_card"])
        slider_row.pack(fill=tk.X, pady=1)
        
        self.speed_slider = tk.Scale(
            slider_row, from_=0.1, to=4.0, resolution=0.1, orient=tk.HORIZONTAL,
            length=230, highlightthickness=0, command=self.on_speed_change, 
            bg=self.colors["bg_card"], fg="#ffffff", troughcolor="#12151b", activebackground=self.colors["accent"]
        )
        self.speed_slider.set(1.0)
        self.speed_slider.pack(side=tk.LEFT, padx=4)
        
        self.lbl_speed_val = tk.Label(slider_row, text="1.0x", bg=self.colors["bg_card"], fg=self.colors["text_primary"], font=("Segoe UI", 9))
        self.lbl_speed_val.pack(side=tk.LEFT, padx=4)
        
        self.btn_reset_speed = ttk.Button(slider_row, text=self.t("reset_speed"), command=lambda: self.speed_slider.set(1.0), style="Modern.TButton")
        self.btn_reset_speed.pack(side=tk.LEFT, padx=4)
        
        skip_row = tk.Frame(self.frame_speed, bg=self.colors["bg_card"])
        skip_row.pack(fill=tk.X, pady=2)
        
        self.chk_skip = ttk.Checkbutton(skip_row, text=self.t("skip_frames"), variable=self.var_enable_skipping, command=self.update_skip_prediction)
        self.chk_skip.pack(side=tk.LEFT, padx=4)
        
        self.lbl_prediction = tk.Label(skip_row, text=self.t("predict_single"), bg=self.colors["bg_card"], fg=self.colors["accent"], font=("Segoe UI", 8, "bold"))
        self.lbl_prediction.pack(side=tk.LEFT, padx=6)

        # 畫布
        self.canvas = tk.Canvas(self.left_frame, width=640, height=410, bg=self.colors["canvas_bg"], highlightthickness=1, highlightbackground=self.colors["bg_card_border"])
        self.canvas.pack(side=tk.TOP, pady=4)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
        
        self.lbl_scale_info = tk.Label(self.left_frame, text=self.t("scale_not_set"), bg=self.colors["bg_main"], fg="#38bdf8", font=("Segoe UI", 9, "bold"))
        self.lbl_scale_info.pack(side=tk.TOP, anchor=tk.W, pady=1)

        bottom_bar = tk.Frame(self.left_frame, bg=self.colors["bg_main"])
        bottom_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=2)
        
        self.btn_info = ttk.Button(bottom_bar, text=self.t("info_btn"), command=self.show_about_info, style="Modern.TButton")
        self.btn_info.pack(side=tk.LEFT, padx=2)
        
        self.lbl_status = tk.Label(bottom_bar, text=self.t("status_ready"), bg=self.colors["bg_main"], fg="#60a5fa", font=("Segoe UI", 9))
        self.lbl_status.pack(side=tk.LEFT, padx=8)

        # 右側圖表面板
        self.right_frame = tk.Frame(main_workspace, bg=self.colors["bg_main"])
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.checkbox_panel = ttk.LabelFrame(self.right_frame, text=self.t("metrics_panel"), padding=6)
        self.checkbox_panel.pack(side=tk.TOP, fill=tk.X, padx=4, pady=2)
        
        control_subrow = tk.Frame(self.checkbox_panel, bg=self.colors["bg_card"])
        control_subrow.pack(side=tk.TOP, fill=tk.X, pady=1)
        
        self.chk_smooth = ttk.Checkbutton(control_subrow, text=self.t("smooth_chart"), variable=self.var_smooth_chart, command=self.update_all_charts)
        self.chk_smooth.pack(side=tk.LEFT, padx=4)
        
        self.metrics_grid = tk.Frame(self.checkbox_panel, bg=self.colors["bg_card"])
        self.metrics_grid.pack(side=tk.TOP, fill=tk.X, pady=(4, 0))
        
        default_checked = ["Coord_X", "Coord_Y", "Speed", "Accel_Mag", "Path_Length"]
        col_idx, row_idx = 0, 0
        for key in self.metric_colors.keys():
            var = tk.BooleanVar(value=(key in default_checked))
            self.metric_vars[key] = var
            chk = ttk.Checkbutton(self.metrics_grid, text=key, variable=var, command=self.rebuild_chart_container)
            chk.grid(row=row_idx, column=col_idx, sticky="w", padx=4, pady=1)
            self.metric_checkboxes[key] = chk
            col_idx += 1
            if col_idx >= 4:
                col_idx = 0
                row_idx += 1

        chart_scroll_frame = tk.Frame(self.right_frame, bg=self.colors["bg_main"])
        chart_scroll_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=4, pady=4)
        
        self.plot_canvas = tk.Canvas(chart_scroll_frame, bg="#1a1d24", highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(chart_scroll_frame, orient="vertical", command=self.plot_canvas.yview)
        self.scrollable_inner_frame = tk.Frame(self.plot_canvas, bg="#1a1d24")
        self.scrollable_inner_frame.bind("<Configure>", lambda e: self.plot_canvas.configure(scrollregion=self.plot_canvas.bbox("all")))
        self.plot_canvas.create_window((0, 0), window=self.scrollable_inner_frame, anchor="nw")
        self.plot_canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.plot_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.plot_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def change_language(self, lang_code, force=False):
        if self.current_lang == lang_code and not force: return
        self.current_lang = lang_code
        apply_adaptive_matplotlib_font(lang_code)
        
        self.window.title(self.t("title"))
        self.lbl_app_title.config(text=self.t("title"))
        self.lbl_guide_title.config(text=self.t("guide_title"))
        self.lbl_guide_steps.config(text=self.t("guide_steps"))
        
        self.btn_open.config(text=self.t("open_video"))
        self.btn_scale.config(text=self.t("set_scale"))
        self.btn_select.config(text=self.t("select_target"))
        self.btn_color_pick.config(text=self.t("color_target"))
        self.btn_start.config(text=self.t("track_play"))
        self.btn_fast_start.config(text=self.t("fast_track"))
        self.btn_export.config(text=self.t("export_center"))
        self.btn_info.config(text=self.t("info_btn"))
        self.btn_reset_speed.config(text=self.t("reset_speed"))
        self.btn_mark_a.config(text=self.t("set_start_frame"))
        self.btn_mark_b.config(text=self.t("set_end_frame"))
        
        self.frame_mode_select.config(text=self.t("tracking_mode_label"))
        self.frame_trim.config(text=self.t("trim_panel"))
        self.frame_algo.config(text=self.t("algo_settings"))
        self.lbl_prep.config(text=self.t("enhance_label"))
        self.chk_kalman.config(text=self.t("kalman_enable"))
        
        curr_prep_idx = self.combo_prep.current()
        self.combo_prep.config(values=self.t("prep_options"))
        if curr_prep_idx >= 0: self.combo_prep.current(curr_prep_idx)
            
        self.frame_speed.config(text=self.t("play_sampling"))
        self.chk_skip.config(text=self.t("skip_frames"))
        self.checkbox_panel.config(text=self.t("metrics_panel"))
        self.chk_smooth.config(text=self.t("smooth_chart"))
        
        metric_dict = self.t("metrics")
        for k, chk in self.metric_checkboxes.items():
            if k in metric_dict: chk.config(text=metric_dict[k][0])
                
        if self.pixel_per_unit:
            self.lbl_scale_info.config(text=self.t("scale_set").format(unit=self.unit_name, scale=self.pixel_per_unit))
        else:
            self.lbl_scale_info.config(text=self.t("scale_not_set"))
            
        self.update_skip_prediction()
        self.rebuild_chart_container()

    def show_about_info(self):
        info_text = (
            f"{self.t('title')}\n\n"
            "Key Features:\n"
            "• Dynamic Fluid / Plume Extreme Tip Detection (Bottom-most tip)\n"
            "• Universal Mode Dropdown (Fluid vs Rigid vs Fast KCF)\n"
            "• Predefined Color Palette (HSV adaptive thresholding)\n"
            "• Project-based lifecycle management & Button Auto-reset\n"
            "• Video Trimming & Clip Segment Kinematics Extraction\n"
            "Author: 小揖資訊服務 (Xiao Yi IT Services)"
        )
        messagebox.showinfo(self.t("info_btn"), info_text)

    def _on_mousewheel(self, event):
        if event.widget.winfo_toplevel() == self.window:
            self.plot_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def on_speed_change(self, val):
        speed = float(val)
        self.lbl_speed_val.config(text=f"{speed:.1f}x")
        self.update_skip_prediction()

    def update_skip_prediction(self):
        speed = self.speed_slider.get()
        if not self.var_enable_skipping.get():
            self.lbl_prediction.config(text=self.t("predict_single"), fg=self.colors["accent"])
        else:
            if speed < 1.0:
                self.lbl_prediction.config(text=self.t("predict_slow"), fg="#38bdf8")
            else:
                skip = int(speed)
                sampling_pct = (1.0 / skip) * 100
                msg = self.t("predict_skip").format(skip=skip, pct=sampling_pct)
                self.lbl_prediction.config(text=msg, fg="#f87171" if skip >= 3 else "#4ade80")

    def preprocess_frame(self, frame):
        prep_idx = self.combo_prep.current()
        if prep_idx == 2: return frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prep_idx == 0:
            enhanced = self.clahe.apply(gray)
            return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        elif prep_idx == 1:
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            return cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
        return frame

    def open_video_project(self):
        if self.frame_data:
            confirm = messagebox.askyesno("專案重設提示", self.t("reset_confirm"))
            if not confirm: return

        new_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv *.wmv")])
        if not new_path: return

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.video_path = new_path
        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0 or np.isnan(self.fps): self.fps = 30.0
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.total_frames <= 0: self.total_frames = 1

        self.frame_data.clear()
        self.origin_x = None
        self.origin_y = None
        self.pixel_per_unit = None
        self.unit_name = "px"
        self.tracking = False
        self.selecting = False
        self.color_picking = False
        self.bbox = None
        self.lost_frames_counter = 0
        if self.rect:
            self.canvas.delete(self.rect)
            self.rect = None

        self.trim_start = 0
        self.trim_end = self.total_frames - 1
        self.trim_slider.config(to=self.total_frames - 1)
        self.trim_slider.set(0)
        self.update_trim_info_label()

        ret, self.current_frame = self.cap.read()
        if ret:
            self.first_frame = self.current_frame.copy()
            self.display_frame(self.current_frame)
            self.lbl_status.config(text=self.t("status_loaded").format(fps=self.fps, total=self.total_frames))
            
            self.btn_scale.config(state=tk.NORMAL)
            self.btn_select.config(state=tk.NORMAL)
            self.btn_color_pick.config(state=tk.NORMAL)
            self.btn_mark_a.config(state=tk.NORMAL)
            self.btn_mark_b.config(state=tk.NORMAL)
            
            self.btn_start.config(state=tk.DISABLED)
            self.btn_fast_start.config(state=tk.DISABLED)
            self.btn_export.config(state=tk.DISABLED)
            
            self.lbl_scale_info.config(text=self.t("scale_not_set"))
            self.update_all_charts()
            self.update_skip_prediction()

    def display_frame(self, frame):
        frame_resized = cv2.resize(frame, (640, 410))
        cv2_im = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2_im)
        self.photo = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def on_timeline_seek(self, val):
        if self.cap is None or self.tracking: return
        f_idx = int(val)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame_idx = f_idx
            self.current_frame = frame.copy()
            self.display_frame(self.current_frame)

    def mark_trim_start(self):
        curr = int(self.trim_slider.get())
        if curr >= self.trim_end:
            messagebox.showwarning("剪輯設定", "起點不能大於或等於終點！")
            return
        self.trim_start = curr
        self.update_trim_info_label()

    def mark_trim_end(self):
        curr = int(self.trim_slider.get())
        if curr <= self.trim_start:
            messagebox.showwarning("剪輯設定", "終點不能小於或等於起點！")
            return
        self.trim_end = curr
        self.update_trim_info_label()

    def update_trim_info_label(self):
        s_time = self.trim_start / self.fps
        e_time = self.trim_end / self.fps
        self.lbl_trim_info.config(
            text=f"區間：{self.trim_start} ~ {self.trim_end} 幀 ({s_time:.2f}s ~ {e_time:.2f}s | 共 {self.trim_end - self.trim_start + 1} 幀)"
        )

    # ==========================================
    # 核心：顏色提取與流體極限像素尋找
    # ==========================================
    def open_color_picker_settings(self):
        color_win = tk.Toplevel(self.window)
        color_win.title(self.t("color_modal_title"))
        color_win.geometry("520x420")
        color_win.configure(bg=self.colors["bg_card"])
        
        tk.Label(color_win, text=self.t("color_modal_title"), bg=self.colors["bg_card"], fg="#00f7ff", font=("Segoe UI", 11, "bold")).pack(pady=10)
        
        palette_frame = ttk.LabelFrame(color_win, text=self.t("palette_label"), padding=8)
        palette_frame.pack(fill=tk.X, padx=16, pady=4)
        
        grid_pal = tk.Frame(palette_frame, bg=self.colors["bg_card"])
        grid_pal.pack(fill=tk.X)
        
        c_idx, r_idx = 0, 0
        for name, hsv in self.preset_palette.items():
            btn = tk.Button(
                grid_pal, text=name, bg="#334155", fg="#ffffff", activebackground=self.colors["accent"],
                relief="groove", font=("Segoe UI", 8),
                command=lambda val=hsv, n=name: self.apply_preset_color(val, n, color_win)
            )
            btn.grid(row=r_idx, column=c_idx, padx=3, pady=3, sticky="ew")
            c_idx += 1
            if c_idx >= 4:
                c_idx = 0
                r_idx += 1
        for i in range(4):
            grid_pal.columnconfigure(i, weight=1)

        row_dir = tk.Frame(color_win, bg=self.colors["bg_card"])
        row_dir.pack(fill=tk.X, padx=16, pady=8)
        tk.Label(row_dir, text=self.t("dir_select_label"), bg=self.colors["bg_card"], fg="#ffffff").pack(side=tk.LEFT)
        combo_dir = ttk.Combobox(row_dir, values=self.t("dir_options"), state="readonly", width=22)
        combo_dir.set(self.target_direction)
        combo_dir.pack(side=tk.RIGHT)
        
        row_tol = tk.Frame(color_win, bg=self.colors["bg_card"])
        row_tol.pack(fill=tk.X, padx=16, pady=4)
        tk.Label(row_tol, text=self.t("color_tol_label"), bg=self.colors["bg_card"], fg="#ffffff").pack(side=tk.LEFT)
        scale_tol = tk.Scale(row_tol, from_=10, to=60, orient=tk.HORIZONTAL, bg=self.colors["bg_card"], fg="#ffffff", troughcolor="#12151b")
        scale_tol.set(self.color_tolerance)
        scale_tol.pack(side=tk.RIGHT)
        
        def start_custom_picking():
            self.target_direction = combo_dir.get()
            self.color_tolerance = scale_tol.get()
            color_win.destroy()
            self.color_picking = True
            self.lbl_status.config(text=self.t("status_color_pick"))
            
        ttk.Button(color_win, text="🎯 手動點選畫布顏色進行鎖定", command=start_custom_picking, style="Accent.TButton").pack(side=tk.BOTTOM, fill=tk.X, padx=16, pady=16)

    def apply_preset_color(self, hsv_val, color_name, win):
        self.target_hsv = hsv_val
        win.destroy()
        point_or_bbox = self.detect_target_in_frame(self.current_frame)
        if point_or_bbox:
            self.init_clean_tracker(point_or_bbox)
            self.lbl_status.config(text=f"已按【{color_name}】+【{self.target_direction}】鎖定目標！")
            self.btn_start.config(state=tk.NORMAL)
            self.btn_fast_start.config(state=tk.NORMAL)

    def detect_target_in_frame(self, frame):
        """核心影像偵測：支援流體極限端點提取或剛體質心"""
        orig_h, orig_w, _ = frame.shape
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        h, s, v = self.target_hsv
        tol = self.color_tolerance
        
        # 建立色彩遮罩 (容忍高透光流體與暗部)
        lower = np.array([max(0, h - tol), max(15, s - 90), max(15, v - 90)])
        upper = np.array([min(179, h + tol), min(255, s + 90), min(255, v + 90)])
        
        mask = cv2.inRange(hsv_frame, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) > 20]
        
        if not valid_contours:
            return None
            
        mode_str = self.var_tracking_mode.get()

        # 模式 1：流體最底端極值模式 (專為擴散沉降設計)
        if "流體" in mode_str or "羽流" in mode_str or "下方" in self.target_direction:
            all_pts = np.vstack(valid_contours)
            # 遍歷所有輪廓點，找出全畫面 Y 座標最大的極值像素 (最深點)
            bottom_idx = all_pts[:, :, 1].argmax()
            tip_x, tip_y = all_pts[bottom_idx][0]
            
            box_sz = 24
            bx = max(0, int(tip_x - box_sz // 2))
            by = max(0, int(tip_y - box_sz + 2))
            return {"type": "tip", "point": (float(tip_x), float(tip_y)), "bbox": (bx, by, box_sz, box_sz)}

        # 模式 2：色彩輪廓質心模式
        elif "色彩輪廓" in mode_str:
            largest = max(valid_contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                x, y, w, h = cv2.boundingRect(largest)
                cx, cy = x + w/2.0, y + h/2.0
            x, y, w, h = cv2.boundingRect(largest)
            return {"type": "centroid", "point": (float(cx), float(cy)), "bbox": (x, y, w, h)}

        # 模式 3/4：剛體追蹤器 (CSRT / KCF)
        else:
            candidates = [cv2.boundingRect(c) for c in valid_contours]
            center_x, center_y = orig_w / 2.0, orig_h / 2.0
            def score(bbox):
                x, y, w, h = bbox
                cx, cy = x + w / 2.0, y + h / 2.0
                if "中心" in self.target_direction: return math.hypot(cx - center_x, cy - center_y)
                elif "左" in self.target_direction: return cx
                elif "右" in self.target_direction: return -cx
                return cy
            best = min(candidates, key=score)
            return {"type": "rigid", "point": (best[0] + best[2]/2.0, best[1] + best[3]/2.0), "bbox": best}

    def start_calibration(self):
        if self.cap is None: return
        self.calibrating = True
        self.scale_points = []
        self.lbl_status.config(text=self.t("status_calibrating"))

    def start_selection(self):
        if self.cap is None: return
        self.selecting = True
        self.lbl_status.config(text=self.t("select_target_prompt"))

    def on_mouse_press(self, event):
        orig_h, orig_w, _ = self.current_frame.shape
        real_x = int(event.x * (orig_w / 640))
        real_y = int(event.y * (orig_h / 410))

        if self.color_picking:
            self.color_picking = False
            hsv = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2HSV)
            self.target_hsv = hsv[real_y, real_x]
            
            res = self.detect_target_in_frame(self.current_frame)
            if res:
                self.init_clean_tracker(res)
                self.lbl_status.config(text=self.t("status_locked"))
                self.btn_start.config(state=tk.NORMAL)
                self.btn_fast_start.config(state=tk.NORMAL)
            else:
                messagebox.showwarning("鎖定失敗", "點擊位置未找到足夠連通色彩像素，請嘗試加大色彩公差！")
            return

        if self.calibrating:
            self.scale_points.append((real_x, real_y))
            self.canvas.create_oval(event.x-4, event.y-4, event.x+4, event.y+4, fill="#facc15", tags="scale_dot")
            
            if len(self.scale_points) == 2:
                self.calibrating = False
                p1, p2 = self.scale_points[0], self.scale_points[1]
                pixel_dist = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
                
                real_length = simpledialog.askfloat(self.t("scale_len_title"), self.t("scale_len_prompt"), minvalue=0.0001)
                if real_length:
                    unit = simpledialog.askstring(self.t("scale_unit_title"), self.t("scale_unit_prompt"), initialvalue="mm")
                    self.unit_name = unit if unit else "mm"
                    self.pixel_per_unit = pixel_dist / real_length
                    
                    self.lbl_scale_info.config(text=self.t("scale_set").format(unit=self.unit_name, scale=self.pixel_per_unit))
                    self.lbl_status.config(text=self.t("status_ready"))
                    self.update_all_charts()
                self.canvas.delete("scale_dot")
            return

        if not self.selecting: return
        self.init_x, self.init_y = event.x, event.y
        if self.rect: self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(self.init_x, self.init_y, self.init_x, self.init_y, outline="#00f7ff", width=2)

    def on_mouse_drag(self, event):
        if not self.selecting: return
        self.cur_x, self.cur_y = event.x, event.y
        self.canvas.coords(self.rect, self.init_x, self.init_y, self.cur_x, self.cur_y)

    def on_mouse_release(self, event):
        if not self.selecting: return
        self.selecting = False
        
        x = min(self.init_x, self.cur_x)
        y = min(self.init_y, self.cur_y)
        w = abs(self.init_x - self.cur_x)
        h = abs(self.init_y - self.cur_y)
        if w < 5 or h < 5: return

        orig_h, orig_w, _ = self.current_frame.shape
        bbox = (int(x * (orig_w / 640)), int(y * (orig_h / 410)), int(w * (orig_w / 640)), int(h * (orig_h / 410)))
        
        # 手動框選自動切換至 CSRT 剛體追蹤
        self.var_tracking_mode.set(self.mode_options[1])
        res = {"type": "rigid", "point": (bbox[0] + bbox[2]/2.0, bbox[1] + bbox[3]/2.0), "bbox": bbox}
        self.init_clean_tracker(res)
        self.lbl_status.config(text=self.t("status_locked"))
        self.btn_start.config(state=tk.NORMAL)
        self.btn_fast_start.config(state=tk.NORMAL)

    def init_clean_tracker(self, res_data):
        self.bbox = res_data["bbox"]
        cx, cy = res_data["point"]
        self.kalman.init(cx, cy)
        
        mode_str = self.var_tracking_mode.get()
        if "KCF" in mode_str:
            self.tracker = cv2.legacy.TrackerKCF_create() if hasattr(cv2, 'legacy') else cv2.TrackerKCF_create()
            self.tracker.init(self.preprocess_frame(self.current_frame), self.bbox)
        elif "CSRT" in mode_str:
            self.tracker = cv2.legacy.TrackerCSRT_create() if hasattr(cv2, 'legacy') else cv2.TrackerCSRT_create()
            self.tracker.init(self.preprocess_frame(self.current_frame), self.bbox)
        else:
            self.tracker = None  # 流體/色彩模式直接由 detect_target_in_frame 動態逐幀擷取

        # 畫面上高亮框出
        orig_h, orig_w, _ = self.current_frame.shape
        x, y, w, h = self.bbox
        c_x1 = int(x * (640 / orig_w))
        c_y1 = int(y * (410 / orig_h))
        c_x2 = int((x + w) * (640 / orig_w))
        c_y2 = int((y + h) * (410 / orig_h))
        if self.rect: self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(c_x1, c_y1, c_x2, c_y2, outline="#00f7ff", width=2)

    def process_kinematics(self, cx, cy, frame_idx):
        factor = self.pixel_per_unit if self.pixel_per_unit else 1.0
        phys_x = cx / factor
        phys_y = cy / factor
        curr_time = (frame_idx - self.trim_start) / self.fps
        
        if self.origin_x is None:
            self.origin_x = phys_x
            self.origin_y = phys_y
            
        disp_x = phys_x - self.origin_x
        disp_y = phys_y - self.origin_y
        disp_total = math.hypot(disp_x, disp_y)
        
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
            
            step_px = math.hypot(cx - prev["Raw_X"], cy - prev["Raw_Y"])
            tot_path = prev.get("Path_Length", 0.0) + (math.hypot(dx, dy) if step_px > 2.0 else 0.0)
            
            vel_x = dx / dt
            vel_y = dy / dt
            speed = math.hypot(vel_x, vel_y)
            
            raw_accel_x = (vel_x - prev["Vel_X"]) / dt
            raw_accel_y = (vel_y - prev["Vel_Y"]) / dt
            accel_x = 0.7 * raw_accel_x + 0.3 * prev.get("Accel_X", 0.0)
            accel_y = 0.7 * raw_accel_y + 0.3 * prev.get("Accel_Y", 0.0)
            accel_mag = math.hypot(accel_x, accel_y)
            
            jerk_x = (accel_x - prev["Accel_X"]) / dt
            jerk_y = (accel_y - prev["Accel_Y"]) / dt
            jerk_mag = math.hypot(jerk_x, jerk_y)
            
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
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.trim_start)
        _, self.current_frame = self.cap.read()
        self.frame_data.clear()
        self.origin_x = None
        self.origin_y = None
        self.lost_frames_counter = 0
        self.update_all_charts()

    def start_tracking(self):
        if self.bbox is None: return
        self.prepare_for_reanalysis()
        self.tracking = True
        self.set_buttons_state(tk.DISABLED)
        self.current_frame_idx = self.trim_start
        self.update_loop()

    def start_fast_tracking(self):
        if self.bbox is None: return
        self.prepare_for_reanalysis()
        self.set_buttons_state(tk.DISABLED)
        self.lbl_status.config(text=self.t("status_fast_running"))
        self.window.update()
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.trim_start)
        curr_f = self.trim_start
        mode_str = self.var_tracking_mode.get()
        
        while curr_f <= self.trim_end:
            ret, frame = self.cap.read()
            if not ret: break
            
            cx, cy = None, None
            if "流體" in mode_str or "色彩輪廓" in mode_str:
                res = self.detect_target_in_frame(frame)
                if res:
                    self.lost_frames_counter = 0
                    cx, cy = res["point"]
                else:
                    self.lost_frames_counter += 1
                    if self.lost_frames_counter >= self.lost_frames_limit: break
            else:
                proc_frame = self.preprocess_frame(frame)
                success, bbox = self.tracker.update(proc_frame)
                if success:
                    self.lost_frames_counter = 0
                    x, y, w, h = [int(v) for v in bbox]
                    cx, cy = x + w/2.0, y + h/2.0
                else:
                    self.lost_frames_counter += 1
                    if self.lost_frames_counter >= self.lost_frames_limit: break

            if cx is not None and cy is not None:
                row = self.process_kinematics(cx, cy, curr_f)
                self.frame_data.append(row)
            curr_f += 1
        
        self.update_all_charts()
        self.end_tracking(self.t("status_tracking_done"))

    def update_loop(self):
        if not self.tracking: return
        
        speed_factor = self.speed_slider.get()
        if speed_factor <= 0: speed_factor = 1.0
        skip_frames = int(speed_factor) if (self.var_enable_skipping.get() and speed_factor >= 1.0) else 1
            
        ret, frame = False, None
        for _ in range(skip_frames):
            if self.current_frame_idx > self.trim_end: break
            ret, frame = self.cap.read()
            if not ret: break
            self.current_frame_idx += 1
            
        if not ret or frame is None or self.current_frame_idx > self.trim_end:
            self.update_all_charts()
            self.end_tracking(self.t("status_tracking_done"))
            return
            
        self.trim_slider.set(self.current_frame_idx)
        self.current_frame = frame.copy()
        mode_str = self.var_tracking_mode.get()
        cx, cy = None, None

        # 執行分流追蹤
        if "流體" in mode_str or "色彩輪廓" in mode_str:
            res = self.detect_target_in_frame(frame)
            if res:
                self.lost_frames_counter = 0
                cx, cy = res["point"]
                x, y, w, h = [int(v) for v in res["bbox"]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                self.lost_frames_counter += 1
                if self.lost_frames_counter >= self.lost_frames_limit:
                    self.update_all_charts()
                    self.end_tracking(self.t("status_lost"))
                    return
        else:
            proc_frame = self.preprocess_frame(frame)
            success, bbox = self.tracker.update(proc_frame)
            if success:
                self.lost_frames_counter = 0
                x, y, w, h = [int(v) for v in bbox]
                raw_cx, raw_cy = x + w/2.0, y + h/2.0
                if self.var_use_kalman.get():
                    self.kalman.predict()
                    cx, cy = self.kalman.correct(raw_cx, raw_cy)
                else: cx, cy = raw_cx, raw_cy
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                self.lost_frames_counter += 1
                if self.var_use_kalman.get() and self.lost_frames_counter < 15:
                    cx, cy = self.kalman.predict()
                    cv2.circle(frame, (int(cx), int(cy)), 10, (0, 165, 255), 2)
                elif self.lost_frames_counter >= self.lost_frames_limit:
                    self.update_all_charts()
                    self.end_tracking(self.t("status_lost"))
                    return

        if cx is not None and cy is not None:
            row = self.process_kinematics(cx, cy, self.current_frame_idx)
            self.frame_data.append(row)
            
            if len(self.frame_data) > 1:
                trail_len = min(40, len(self.frame_data))
                for i in range(len(self.frame_data) - trail_len + 1, len(self.frame_data)):
                    pt1 = (int(self.frame_data[i-1]["Raw_X"]), int(self.frame_data[i-1]["Raw_Y"]))
                    pt2 = (int(self.frame_data[i]["Raw_X"]), int(self.frame_data[i]["Raw_Y"]))
                    cv2.line(frame, pt1, pt2, (0, 220, 255), 2, cv2.LINE_AA)
            
            # 最前端點以鮮紅實心圓標示
            cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"v: {row['Speed']:.1f} {self.unit_name}/s | Tip Y: {cy:.0f}px", 
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            
            if len(self.frame_data) % 5 == 0:
                self.update_all_charts()
            
        self.display_frame(frame)
        delay_ms = max(1, int((1000.0 / self.fps) / speed_factor)) if (not self.var_enable_skipping.get() or speed_factor < 1.0) else 1
        self.window.after(delay_ms, self.update_loop)

    def rebuild_chart_container(self):
        for widget in self.scrollable_inner_frame.winfo_children():
            widget.destroy()
        self.chart_cards.clear()
        
        selected_metrics = [k for k, v in self.metric_vars.items() if v.get()]
        if not selected_metrics: return
            
        metric_dict = self.t("metrics")
        for key in selected_metrics:
            label_name, unit_pattern = metric_dict.get(key, (key, "{unit}"))
            color = self.metric_colors.get(key, "#2563eb")
            
            card_frame = tk.Frame(self.scrollable_inner_frame, bg="#ffffff", bd=1, relief="ridge")
            card_frame.pack(fill=tk.X, expand=True, padx=4, pady=4)
            
            fig, ax = plt.subplots(figsize=(5.6, 1.8), facecolor='#ffffff')
            fig.subplots_adjust(left=0.20, right=0.96, top=0.88, bottom=0.26)
            
            canvas = FigureCanvasTkAgg(fig, master=card_frame)
            canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            btn_view = ttk.Button(card_frame, text=self.t("view_data"), style="Modern.TButton", command=lambda k=key: self.open_single_metric_viewer(k))
            btn_view.pack(side=tk.RIGHT, padx=8, pady=10)
            
            self.chart_cards[key] = {"fig": fig, "ax": ax, "canvas": canvas, "color": color, "label": label_name, "unit": unit_pattern}
        self.update_all_charts()

    def update_all_charts(self):
        df = pd.DataFrame(self.frame_data) if self.frame_data else pd.DataFrame()
        smooth_active = self.var_smooth_chart.get()
        
        for key, item in self.chart_cards.items():
            ax = item["ax"]
            ax.cla()
            ax.grid(True, linestyle="--", alpha=0.4)
            
            actual_unit = item["unit"].format(unit=self.unit_name)
            ax.set_ylabel(f"{item['label']}\n({actual_unit})", fontsize=8, labelpad=8)
            ax.set_xlabel(self.t("time_axis"), fontsize=8)
            ax.tick_params(axis='both', labelsize=8)
            
            if not df.empty and key in df.columns:
                t = df["Time"].values
                y = df[key].values
                if smooth_active and len(t) >= 4 and len(np.unique(t)) == len(t):
                    try:
                        t_dense = np.linspace(t.min(), t.max(), min(len(t) * 4, 300))
                        spl = make_interp_spline(t, y, k=3)
                        ax.plot(t_dense, spl(t_dense), color=item["color"], linewidth=1.6)
                    except Exception:
                        ax.plot(t, y, color=item["color"], linewidth=1.5)
                else:
                    ax.plot(t, y, color=item["color"], linewidth=1.5)
            item["canvas"].draw()

    def open_single_metric_viewer(self, metric_key):
        if not self.frame_data:
            messagebox.showwarning(self.t("title"), self.t("no_data_warn"))
            return
            
        df = pd.DataFrame(self.frame_data)
        metric_dict = self.t("metrics")
        label_name, unit_pattern = metric_dict.get(metric_key, (metric_key, "{unit}"))
        actual_unit = unit_pattern.format(unit=self.unit_name)
        
        sub_win = tk.Toplevel(self.window)
        sub_win.title(f"{label_name} - {self.t('view_data')}")
        sub_win.geometry("560x480")
        sub_win.configure(bg=self.colors["bg_main"])
        
        top_bar = tk.Frame(sub_win, bg=self.colors["bg_card"], padx=8, pady=8)
        top_bar.pack(fill=tk.X)
        
        val_series = df[metric_key]
        summary_text = self.t("summary_fmt").format(n=len(df), max_v=val_series.max(), unit=actual_unit, mean_v=val_series.mean())
        tk.Label(top_bar, text=summary_text, bg=self.colors["bg_card"], fg="#00f7ff", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
        
        tree_frame = tk.Frame(sub_win, bg=self.colors["bg_main"])
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
        
        tree = ttk.Treeview(tree_frame, columns=("Frame", "Time", "Value"), show="headings", height=15)
        tree.heading("Frame", text=self.t("frame_col"))
        tree.heading("Time", text=self.t("time_col"))
        tree.heading("Value", text=f"{label_name} ({actual_unit})")
        tree.column("Frame", width=100, anchor="center")
        tree.column("Time", width=120, anchor="center")
        tree.column("Value", width=200, anchor="center")
        
        tree_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=tree_scroll.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        for _, row in df.iterrows():
            tree.insert("", "end", values=(int(row["Frame"]), f"{row['Time']:.3f}", f"{row[metric_key]:.4f}"))

    def open_advanced_export_dialog(self):
        if not self.frame_data:
            messagebox.showwarning(self.t("title"), self.t("no_data_warn"))
            return
            
        export_win = tk.Toplevel(self.window)
        export_win.title(self.t("export_modal_title"))
        export_win.geometry("560x640")
        export_win.resizable(False, False)
        export_win.configure(bg=self.colors["bg_card"])
        
        tk.Label(export_win, text=self.t("export_modal_title"), bg=self.colors["bg_card"], fg="#00f7ff", font=("Segoe UI", 12, "bold")).pack(pady=(12, 6))
        
        sec1 = ttk.LabelFrame(export_win, text=self.t("export_step1"), padding=8)
        sec1.pack(fill=tk.X, padx=12, pady=4)
        
        check_inner = tk.Frame(sec1, bg=self.colors["bg_card"])
        check_inner.pack(fill=tk.BOTH, expand=True)
        
        export_vars = {}
        metric_dict = self.t("metrics")
        col_idx, row_idx = 0, 0
        for key in self.metric_colors.keys():
            label_text = metric_dict.get(key, (key, ""))[0]
            initial_val = self.metric_vars.get(key, tk.BooleanVar(value=True)).get()
            var = tk.BooleanVar(value=initial_val)
            export_vars[key] = var
            chk = ttk.Checkbutton(check_inner, text=label_text, variable=var)
            chk.grid(row=row_idx, column=col_idx, sticky="w", padx=4, pady=2)
            col_idx += 1
            if col_idx >= 2:
                col_idx = 0
                row_idx += 1
                
        btn_sel_frame = tk.Frame(sec1, bg=self.colors["bg_card"])
        btn_sel_frame.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(btn_sel_frame, text=self.t("select_all"), command=lambda: [v.set(True) for v in export_vars.values()], style="Modern.TButton").pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_sel_frame, text=self.t("deselect_all"), command=lambda: [v.set(False) for v in export_vars.values()], style="Modern.TButton").pack(side=tk.LEFT, padx=2)
        
        sec2 = ttk.LabelFrame(export_win, text=self.t("export_step2"), padding=8)
        sec2.pack(fill=tk.X, padx=12, pady=6)
        
        fmt_row = tk.Frame(sec2, bg=self.colors["bg_card"])
        fmt_row.pack(fill=tk.X, pady=4)
        tk.Label(fmt_row, text=self.t("export_format_label"), bg=self.colors["bg_card"], fg="#ffffff", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
        
        var_format = tk.StringVar(value="PNG")
        for fmt, desc in [("PNG", "PNG [點陣]"), ("PDF", "PDF [向量]"), ("SVG", "SVG [向量]"), ("JPG", "JPG [點陣]")]:
            ttk.Radiobutton(fmt_row, text=desc, value=fmt, variable=var_format).pack(side=tk.LEFT, padx=5)

        var_combine = tk.BooleanVar(value=True)
        chk_combine = ttk.Checkbutton(sec2, text=self.t("export_combine_label"), variable=var_combine)
        chk_combine.pack(anchor=tk.W, pady=4)
        
        var_include_csv = tk.BooleanVar(value=True)
        chk_csv = ttk.Checkbutton(sec2, text=self.t("export_include_csv"), variable=var_include_csv)
        chk_csv.pack(anchor=tk.W, pady=4)

        def execute_advanced_export():
            chosen = [k for k, v in export_vars.items() if v.get()]
            if not chosen:
                messagebox.showwarning(self.t("title"), "請至少勾選一項數據！")
                return
                
            fmt = var_format.get().lower()
            combine = var_combine.get()
            with_csv = var_include_csv.get()
            
            save_path = filedialog.asksaveasfilename(defaultextension=f".{fmt}", filetypes=[(f"{fmt.upper()} File", f"*.{fmt}")])
            if not save_path: return

            base_no_ext, _ = os.path.splitext(save_path)
            df = pd.DataFrame(self.frame_data)
            t = df["Time"].values
            smooth_active = self.var_smooth_chart.get()
            metric_info = self.t("metrics")

            if combine:
                num_plots = len(chosen)
                cols = 2 if num_plots <= 6 else 3
                rows = math.ceil(num_plots / cols)
                fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 3.2 * rows), facecolor='#ffffff')
                fig.subplots_adjust(hspace=0.45, wspace=0.35)
                flat_axes = np.array(axes).reshape(-1)
                
                for idx, key in enumerate(chosen):
                    ax = flat_axes[idx]
                    ax.grid(True, linestyle="--", alpha=0.5)
                    label_name, unit_pattern = metric_info.get(key, (key, "{unit}"))
                    actual_unit = unit_pattern.format(unit=self.unit_name)
                    color = self.metric_colors.get(key, "#2563eb")
                    y = df[key].values
                    
                    if smooth_active and len(t) >= 4 and len(np.unique(t)) == len(t):
                        try:
                            t_dense = np.linspace(t.min(), t.max(), min(len(t) * 4, 300))
                            spl = make_interp_spline(t, y, k=3)
                            ax.plot(t_dense, spl(t_dense), color=color, linewidth=1.8)
                        except Exception:
                            ax.plot(t, y, color=color, linewidth=1.5)
                    else:
                        ax.plot(t, y, color=color, linewidth=1.5)
                        
                    ax.set_title(label_name, fontsize=10, fontweight="bold")
                    ax.set_xlabel(self.t("time_axis"), fontsize=9)
                    ax.set_ylabel(f"({actual_unit})", fontsize=9)
                    ax.tick_params(axis='both', labelsize=8)

                for empty_idx in range(num_plots, len(flat_axes)):
                    flat_axes[empty_idx].axis('off')
                fig.tight_layout()
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
            else:
                for key in chosen:
                    fig, ax = plt.subplots(figsize=(7, 4), facecolor='#ffffff')
                    ax.grid(True, linestyle="--", alpha=0.5)
                    label_name, unit_pattern = metric_info.get(key, (key, "{unit}"))
                    actual_unit = unit_pattern.format(unit=self.unit_name)
                    color = self.metric_colors.get(key, "#2563eb")
                    y = df[key].values
                    
                    if smooth_active and len(t) >= 4 and len(np.unique(t)) == len(t):
                        try:
                            t_dense = np.linspace(t.min(), t.max(), min(len(t) * 4, 300))
                            spl = make_interp_spline(t, y, k=3)
                            ax.plot(t_dense, spl(t_dense), color=color, linewidth=2.0)
                        except Exception:
                            ax.plot(t, y, color=color, linewidth=1.6)
                    else:
                        ax.plot(t, y, color=color, linewidth=1.6)
                        
                    ax.set_title(label_name, fontsize=11, fontweight="bold")
                    ax.set_xlabel(self.t("time_axis"), fontsize=10)
                    ax.set_ylabel(f"{label_name} ({actual_unit})", fontsize=10)
                    fig.tight_layout()
                    fig.savefig(f"{base_no_ext}_{key}.{fmt}", dpi=300, bbox_inches='tight')
                    plt.close(fig)

            if with_csv:
                export_cols = ["Frame", "Time"] + chosen
                df[export_cols].to_csv(f"{base_no_ext}.csv", index=False)

            export_win.destroy()
            messagebox.showinfo(self.t("title"), self.t("export_success").format(path=save_path))

        ttk.Button(export_win, text=self.t("btn_export_action"), command=execute_advanced_export, style="Accent.TButton").pack(side=tk.BOTTOM, fill=tk.X, padx=12, pady=12)

    def set_buttons_state(self, state):
        self.btn_open.config(state=state)
        self.btn_scale.config(state=state)
        self.btn_select.config(state=state)
        self.btn_color_pick.config(state=state)
        self.btn_start.config(state=state)
        self.btn_fast_start.config(state=state)
        self.btn_mark_a.config(state=state)
        self.btn_mark_b.config(state=state)

    def end_tracking(self, message):
        self.tracking = False
        self.set_buttons_state(tk.NORMAL)
        self.btn_export.config(state=tk.NORMAL)
        self.lbl_status.config(text=message)

if __name__ == "__main__":
    root = tk.Tk()
    app = UniversalPhysicsTrackerApp(root)
    root.mainloop()