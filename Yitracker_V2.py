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

# ==========================================
# 1. 國際化多國語言字典 (i18n)
# ==========================================
TRANSLATIONS = {
    "zh_TW": {
        "title": "科研級運動追蹤分析工具 V3.5 Pro",
        "open_video": "📁 開啟影片",
        "set_scale": "📐 設定比例尺",
        "select_target": "🎯 框選目標",
        "track_play": "▶ 播放追蹤",
        "fast_track": "⚡ 快速解算",
        "export_csv": "💾 匯出 CSV",
        "algo_settings": "⚙️ 演算法與影像設定",
        "algo_label": "核心：",
        "enhance_label": "影像增強：",
        "kalman_enable": "啟用卡爾曼濾波 (抗遮擋與平滑)",
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
        "status_ready": "狀態：就緒，請載入影片",
        "status_loaded": "狀態：影片已載入 (FPS: {fps:.2f})",
        "status_calibrating": "比例尺模式：請在畫面上「點選兩點」已知長度",
        "status_locked": "目標已鎖定！可點選播放追蹤或快速解算",
        "status_tracking_done": "影片結束，分析完成！",
        "status_lost": "目標遺失超時，追蹤結束。",
        "status_fast_running": "⚡ 正在以極速計算全量科學數據...",
        "select_target_prompt": "請在畫面拖曳滑鼠框選目標物",
        "scale_len_title": "比例尺長度",
        "scale_len_prompt": "請輸入這兩點之間的物理長度：",
        "scale_unit_title": "單位設定",
        "scale_unit_prompt": "請輸入物理長度單位 (例: mm, cm, m):",
        "no_data_warn": "目前尚無任何追蹤數據！",
        "export_title": "匯出 CSV - 自選物理量",
        "export_prompt": "請勾選要匯出至 CSV 的科學計量：",
        "select_all": "全選",
        "deselect_all": "全不選",
        "confirm_export": "確認匯出",
        "export_success": "數據已儲存！共匯出 {count} 項物理量至：\n{path}",
        "time_axis": "時間 Time (s)",
        "frame_col": "影格 (Frame)",
        "time_col": "時間 (Time s)",
        "summary_fmt": "樣本數：{n} | 最大值：{max_v:.2f} {unit} | 平均值：{mean_v:.2f} {unit}",
        "prep_options": ["CLAHE (高對比自適應)", "灰階去噪 (高斯濾波)", "原始色彩 (無處理)"],
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
    "zh_CN": {
        "title": "科研级运动追踪分析工具 V3.5 Pro",
        "open_video": "📁 打开视频",
        "set_scale": "📐 设定标尺",
        "select_target": "🎯 框选目标",
        "track_play": "▶ 播放追踪",
        "fast_track": "⚡ 快速解算",
        "export_csv": "💾 导出 CSV",
        "algo_settings": "⚙️ 算法与图像设定",
        "algo_label": "核心：",
        "enhance_label": "图像增强：",
        "kalman_enable": "启用卡尔曼滤波 (抗遮挡与平滑)",
        "play_sampling": "⏩ 播放与采样控制",
        "reset_speed": "重置 1.0x",
        "skip_frames": "启用跳帧加速 (倍速时生效)",
        "predict_single": "[逐帧解算模式 (100% 采样)]",
        "predict_slow": "[慢动作模式 (延时解算)]",
        "predict_skip": "[跳帧解算：每 {skip} 帧计算一次 (约 {pct:.0f}% 采样)]",
        "scale_not_set": "标尺：未设定 (px)",
        "scale_set": "标尺：1 {unit} = {scale:.2f} px",
        "smooth_chart": "🌊 启用图表曲线平滑化 (Spline)",
        "metrics_panel": "🔬 勾选显示的物理量 (实时绘图 / 独立查看)",
        "view_data": "📊 查看数据",
        "info_btn": "ℹ️ 关于",
        "status_ready": "状态：就绪，请加载视频",
        "status_loaded": "状态：视频已加载 (FPS: {fps:.2f})",
        "status_calibrating": "标尺模式：请在画面上“点击两点”已知长度",
        "status_locked": "目标已锁定！可点击播放追踪或快速解算",
        "status_tracking_done": "视频播放结束，分析完成！",
        "status_lost": "目标丢失超时，追踪结束。",
        "status_fast_running": "⚡ 正在极速计算全量科学数据...",
        "select_target_prompt": "请在画面拖曳鼠标框选目标物",
        "scale_len_title": "标尺长度",
        "scale_len_prompt": "请输入这两点之间的物理长度：",
        "scale_unit_title": "单位设定",
        "scale_unit_prompt": "请输入物理长度单位 (例: mm, cm, m):",
        "no_data_warn": "目前尚无任何追踪数据！",
        "export_title": "导出 CSV - 自选物理量",
        "export_prompt": "请勾选要导出至 CSV 的物理量：",
        "select_all": "全选",
        "deselect_all": "全不选",
        "confirm_export": "确认导出",
        "export_success": "数据已保存！共导出 {count} 项物理量至：\n{path}",
        "time_axis": "时间 Time (s)",
        "frame_col": "帧数 (Frame)",
        "time_col": "时间 (Time s)",
        "summary_fmt": "样本数：{n} | 最大值：{max_v:.2f} {unit} | 平均值：{mean_v:.2f} {unit}",
        "prep_options": ["CLAHE (高对比自适应)", "灰度去噪 (高斯滤波)", "原始色彩 (无处理)"],
        "metrics": {
            "Coord_X": ("X 轴坐标 (X)", "{unit}"),
            "Coord_Y": ("Y 轴坐标 (Y)", "{unit}"),
            "Disp_X": ("X 轴位移 (ΔX)", "{unit}"),
            "Disp_Y": ("Y 轴位移 (ΔY)", "{unit}"),
            "Disp_Total": ("总位移量 (|Δr|)", "{unit}"),
            "Path_Length": ("累计路径长", "{unit}"),
            "Vel_X": ("X 轴速度分量 (Vx)", "{unit}/s"),
            "Vel_Y": ("Y 轴速度分量 (Vy)", "{unit}/s"),
            "Speed": ("瞬时速率 (|v|)", "{unit}/s"),
            "Accel_X": ("X 轴加速度 (ax)", "{unit}/s²"),
            "Accel_Y": ("Y 轴加速度 (ay)", "{unit}/s²"),
            "Accel_Mag": ("瞬时加速度大小 (|a|)", "{unit}/s²"),
            "Jerk_Mag": ("瞬时急动度 (|J|)", "{unit}/s³")
        }
    },
    "en": {
        "title": "Scientific Motion Physics Tracker V3.5 Pro",
        "open_video": "📁 Open Video",
        "set_scale": "📐 Set Scale",
        "select_target": "🎯 Select ROI",
        "track_play": "▶ Track & Play",
        "fast_track": "⚡ Fast Solve",
        "export_csv": "💾 Export CSV",
        "algo_settings": "⚙️️ Tracking & Vision Settings",
        "algo_label": "Algorithm:",
        "enhance_label": "Enhance:",
        "kalman_enable": "Enable Kalman Filter (Occlusion & Smooth)",
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
        "status_ready": "Status: Ready. Load a video file to begin.",
        "status_loaded": "Status: Video loaded (FPS: {fps:.2f})",
        "status_calibrating": "Calibration: Click 2 points with known physical distance.",
        "status_locked": "Target locked. Ready for Track & Play or Fast Solve.",
        "status_tracking_done": "Tracking complete. Video ended.",
        "status_lost": "Target lost (timeout exceeded).",
        "status_fast_running": "⚡ Computing full kinematic parameters at high speed...",
        "select_target_prompt": "Drag mouse over the canvas to select the target.",
        "scale_len_title": "Scale Distance",
        "scale_len_prompt": "Enter the real physical distance between points:",
        "scale_unit_title": "Unit Settings",
        "scale_unit_prompt": "Enter unit name (e.g., mm, cm, m):",
        "no_data_warn": "No tracking data available yet!",
        "export_title": "Export CSV - Select Metrics",
        "export_prompt": "Select physical metrics to include in CSV export:",
        "select_all": "Select All",
        "deselect_all": "Deselect All",
        "confirm_export": "Export CSV",
        "export_success": "Data saved! Successfully exported {count} metrics to:\n{path}",
        "time_axis": "Time (s)",
        "frame_col": "Frame #",
        "time_col": "Time (s)",
        "summary_fmt": "Samples: {n} | Max: {max_v:.2f} {unit} | Mean: {mean_v:.2f} {unit}",
        "prep_options": ["CLAHE (Adaptive)", "Gaussian Denoise", "Raw Color"],
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
    },
    "ja": {
        "title": "科学運動追跡・力学解析システム V3.5 Pro",
        "open_video": "📁 動画を開く",
        "set_scale": "📐 スケール設定",
        "select_target": "🎯 対象を選択",
        "track_play": "▶ 追跡再生",
        "fast_track": "⚡ 高速解析",
        "export_csv": "💾 CSV出力",
        "algo_settings": "⚙️ アルゴリズムと画像強調",
        "algo_label": "エンジン：",
        "enhance_label": "画像補正：",
        "kalman_enable": "カルマンフィルタ有効化 (遮蔽補正・平滑化)",
        "play_sampling": "⏩ 再生速度とサンプリング",
        "reset_speed": "リセット 1.0x",
        "skip_frames": "フレームスキップ有効 (倍速時)",
        "predict_single": "[全フレーム解析モード (100% 抽出)]",
        "predict_slow": "[スロー再生モード]",
        "predict_skip": "[スキップ解析: {skip} フレーム毎 (約 {pct:.0f}%)]",
        "scale_not_set": "スケール：未設定 (px)",
        "scale_set": "スケール：1 {unit} = {scale:.2f} px",
        "smooth_chart": "🌊 グラフ曲線の平滑化 (スプライン補間)",
        "metrics_panel": "🔬 表示する物理量を選択 (リアルタイム描画)",
        "view_data": "📊 データ表示",
        "info_btn": "ℹ️ バージョン情報",
        "status_ready": "状態：準備完了。動画を読み込んでください。",
        "status_loaded": "状態：動画読込完了 (FPS: {fps:.2f})",
        "status_calibrating": "スケール設定：画面上で既知の長さの2点をクリックしてください。",
        "status_locked": "ターゲット捕捉完了！追跡再生または高速解析を開始できます。",
        "status_tracking_done": "動画終了、解析が完了しました。",
        "status_lost": "ターゲットを見失いました。",
        "status_fast_running": "⚡ 全データを高速解析中...",
        "select_target_prompt": "画面上でマウスをドラッグして追跡対象を囲んでください。",
        "scale_len_title": "物理長の入力",
        "scale_len_prompt": "選択した2点間の実際の長さを入力してください：",
        "scale_unit_title": "単位設定",
        "scale_unit_prompt": "単位を入力してください (例: mm, cm, m):",
        "no_data_warn": "追跡データがまだありません！",
        "export_title": "CSVエクスポート - 項目選択",
        "export_prompt": "出力する物理量を選択してください：",
        "select_all": "すべて選択",
        "deselect_all": "選択解除",
        "confirm_export": "CSVを保存",
        "export_success": "保存完了！ {count} 項目を保存しました：\n{path}",
        "time_axis": "時間 Time (s)",
        "frame_col": "フレーム #",
        "time_col": "時間 (s)",
        "summary_fmt": "標本数: {n} | 最大値: {max_v:.2f} {unit} | 平均値: {mean_v:.2f} {unit}",
        "prep_options": ["CLAHE (適応ヒストグラム)", "ガウス平滑化 (ノイズ除去)", "元画像 (処理なし)"],
        "metrics": {
            "Coord_X": ("X軸座標 (X)", "{unit}"),
            "Coord_Y": ("Y軸座標 (Y)", "{unit}"),
            "Disp_X": ("X軸変位 (ΔX)", "{unit}"),
            "Disp_Y": ("Y軸変位 (ΔY)", "{unit}"),
            "Disp_Total": ("全変位量 (|Δr|)", "{unit}"),
            "Path_Length": ("移動累積距離", "{unit}"),
            "Vel_X": ("X軸速度 (Vx)", "{unit}/s"),
            "Vel_Y": ("Y軸速度 (Vy)", "{unit}/s"),
            "Speed": ("瞬時速さ (|v|)", "{unit}/s"),
            "Accel_X": ("X軸加速度 (ax)", "{unit}/s²"),
            "Accel_Y": ("Y軸加速度 (ay)", "{unit}/s²"),
            "Accel_Mag": ("瞬時加速度 (|a|)", "{unit}/s²"),
            "Jerk_Mag": ("躍度 / ジャーク (|J|)", "{unit}/s³")
        }
    },
    "es": {
        "title": "Rastreador Cinemático y de Movimiento Científico V3.5 Pro",
        "open_video": "📁 Abrir Video",
        "set_scale": "📐 Calibrar Escala",
        "select_target": "🎯 Seleccionar ROI",
        "track_play": "▶ Reproducir y Seguir",
        "fast_track": "⚡ Cálculo Rápido",
        "export_csv": "💾 Exportar CSV",
        "algo_settings": "⚙️ Algoritmo y Procesamiento",
        "algo_label": "Algoritmo:",
        "enhance_label": "Mejora:",
        "kalman_enable": "Filtro de Kalman (Anti-oclusión y Suavizado)",
        "play_sampling": "⏩ Control de Velocidad y Muestreo",
        "reset_speed": "Restablecer 1.0x",
        "skip_frames": "Salto de fotogramas (Velocidad > 1x)",
        "predict_single": "[Modo continuo (100% muestreo)]",
        "predict_slow": "[Cámara lenta]",
        "predict_skip": "[Salto: 1 cálculo cada {skip} fotogramas (~{pct:.0f}%)]",
        "scale_not_set": "Escala: Sin calibrar (px)",
        "scale_set": "Escala: 1 {unit} = {scale:.2f} px",
        "smooth_chart": "🌊 Suavizado de curvas (Spline cúbica)",
        "metrics_panel": "🔬 Variables Cinemáticas a Graficar",
        "view_data": "📊 Ver Datos",
        "info_btn": "ℹ️ Información",
        "status_ready": "Estado: Listo. Cargue un video para comenzar.",
        "status_loaded": "Estado: Video cargado (FPS: {fps:.2f})",
        "status_calibrating": "Calibración: Haga clic en 2 puntos con distancia física conocida.",
        "status_locked": "¡Objetivo bloqueado! Listo para rastrear o cálculo rápido.",
        "status_tracking_done": "Análisis finalizado.",
        "status_lost": "Objetivo perdido (límite superado).",
        "status_fast_running": "⚡ Calculando parámetros cinemáticos a alta velocidad...",
        "select_target_prompt": "Arrastre el cursor en el lienzo para delimitar el objetivo.",
        "scale_len_title": "Longitud de Escala",
        "scale_len_prompt": "Ingrese la distancia física real entre los puntos:",
        "scale_unit_title": "Unidad de Medida",
        "scale_unit_prompt": "Ingrese la unidad física (ej. mm, cm, m):",
        "no_data_warn": "¡No hay datos cinemáticos disponibles!",
        "export_title": "Exportar CSV - Seleccionar Variables",
        "export_prompt": "Marque las variables a incluir en el archivo CSV:",
        "select_all": "Marcar Todo",
        "deselect_all": "Desmarcar Todo",
        "confirm_export": "Guardar CSV",
        "export_success": "¡Datos guardados con éxito! Exportadas {count} métricas a:\n{path}",
        "time_axis": "Tiempo Time (s)",
        "frame_col": "Fotograma #",
        "time_col": "Tiempo (s)",
        "summary_fmt": "Muestras: {n} | Máx: {max_v:.2f} {unit} | Media: {mean_v:.2f} {unit}",
        "prep_options": ["CLAHE (Adaptativo)", "Desruido Gaussiano", "Color Original"],
        "metrics": {
            "Coord_X": ("Coordenada X (X)", "{unit}"),
            "Coord_Y": ("Coordenada Y (Y)", "{unit}"),
            "Disp_X": ("Desplazamiento X (ΔX)", "{unit}"),
            "Disp_Y": ("Desplazamiento Y (ΔY)", "{unit}"),
            "Disp_Total": ("Desplazamiento Neto (|Δr|)", "{unit}"),
            "Path_Length": ("Longitud de Trayectoria", "{unit}"),
            "Vel_X": ("Velocidad X (Vx)", "{unit}/s"),
            "Vel_Y": ("Velocidad Y (Vy)", "{unit}/s"),
            "Speed": ("Rapidez Instantánea (|v|)", "{unit}/s"),
            "Accel_X": ("Aceleración X (ax)", "{unit}/s²"),
            "Accel_Y": ("Aceleración Y (ay)", "{unit}/s²"),
            "Accel_Mag": ("Magnitud Aceleración (|a|)", "{unit}/s²"),
            "Jerk_Mag": ("Tirón / Jerk (|J|)", "{unit}/s³")
        }
    }
}

# ==========================================
# 2. 跨平台字型適配
# ==========================================
def apply_adaptive_matplotlib_font(lang_code="zh_TW"):
    sys_name = platform.system()
    if lang_code in ["zh_TW", "zh_CN"]:
        if sys_name == "Windows":
            fonts = ['Microsoft JhengHei', 'SimHei', 'Segoe UI', 'DejaVu Sans']
        elif sys_name == "Darwin":
            fonts = ['PingFang TC', 'PingFang SC', 'Heiti TC', 'Arial Unicode MS', 'DejaVu Sans']
        else:
            fonts = ['Noto Sans CJK TC', 'WenQuanYi Micro Hei', 'DejaVu Sans']
    elif lang_code == "ja":
        if sys_name == "Windows":
            fonts = ['Meiryo', 'MS Gothic', 'Yu Gothic', 'DejaVu Sans']
        elif sys_name == "Darwin":
            fonts = ['Hiragino Sans', 'Apple Gothic', 'DejaVu Sans']
        else:
            fonts = ['Noto Sans CJK JP', 'TakaoPGothic', 'DejaVu Sans']
    else:  # en, es
        fonts = ['Segoe UI', 'Helvetica Neue', 'Arial', 'DejaVu Sans']
        
    plt.rcParams['font.sans-serif'] = fonts
    plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 3. 卡爾曼濾波器 (平滑與短暫遮擋預測)
# ==========================================
class UniversalKalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.04
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.4
        self.is_initialized = False

    def init(self, cx, cy):
        self.kf.statePre = np.array([[float(cx)], [float(cy)], [0.0], [0.0]], np.float32)
        self.kf.statePost = np.array([[float(cx)], [float(cy)], [0.0], [0.0]], np.float32)
        self.is_initialized = True

    def predict(self):
        if not self.is_initialized:
            return 0.0, 0.0
        pred = self.kf.predict()
        return float(pred[0, 0]), float(pred[1, 0])

    def correct(self, cx, cy):
        measurement = np.array([[float(cx)], [float(cy)]], np.float32)
        est = self.kf.correct(measurement)
        return float(est[0, 0]), float(est[1, 0])

# ==========================================
# 4. 主程式介面
# ==========================================
class UniversalPhysicsTrackerApp:
    def __init__(self, window):
        self.window = window
        self.window.geometry("1560x960")
        self.window.minsize(1280, 800)
        self.current_lang = "zh_TW"
        
        apply_adaptive_matplotlib_font(self.current_lang)
        self.window.title(self.t("title"))
        
        # 配色主題 (Modern Slate & Dark Navy)
        self.colors = {
            "bg_main": "#1e222d",
            "bg_card": "#262b36",
            "bg_card_border": "#363c4d",
            "text_primary": "#ffffff",
            "text_secondary": "#a0a8b7",
            "accent": "#00adb5",
            "accent_hover": "#00cbd4",
            "canvas_bg": "#12151b",
            "plot_bg": "#ffffff"
        }
        self.window.configure(bg=self.colors["bg_main"])
        
        # 核心數據狀態
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
        
        # 變數宣告
        self.var_enable_skipping = tk.BooleanVar(value=True)
        self.var_smooth_chart = tk.BooleanVar(value=True)
        self.var_preprocess_mode = tk.StringVar()
        self.var_tracker_type = tk.StringVar(value="CSRT")
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
        try:
            self.style.theme_use("clam")
        except Exception:
            pass
            
        self.style.configure(".", background=self.colors["bg_main"], foreground=self.colors["text_primary"])
        self.style.configure("TFrame", background=self.colors["bg_main"])
        self.style.configure("Card.TFrame", background=self.colors["bg_card"], relief="solid", borderwidth=1)
        self.style.configure("CardLabel.TLabel", background=self.colors["bg_card"], foreground=self.colors["text_primary"], font=("Segoe UI", 9, "bold"))
        self.style.configure("TLabel", background=self.colors["bg_main"], foreground=self.colors["text_primary"], font=("Segoe UI", 9))
        self.style.configure("TCheckbutton", background=self.colors["bg_card"], foreground=self.colors["text_primary"], font=("Segoe UI", 9))
        self.style.map("TCheckbutton", background=[('active', self.colors["bg_card"])])
        
        # 標題框架樣式
        self.style.configure("TLabelframe", background=self.colors["bg_card"], bordercolor=self.colors["bg_card_border"], relief="solid")
        self.style.configure("TLabelframe.Label", background=self.colors["bg_card"], foreground=self.colors["accent"], font=("Segoe UI", 10, "bold"))

        # 現代化按鈕樣式
        self.style.configure(
            "Modern.TButton", 
            background="#3b4252", 
            foreground="#ffffff", 
            font=("Segoe UI", 9, "bold"),
            padding=(8, 4),
            relief="flat",
            borderwidth=0
        )
        self.style.map("Modern.TButton",
            background=[('pressed', '#2e3440'), ('active', '#4c566a'), ('disabled', '#2a2e39')],
            foreground=[('disabled', '#6b7280')]
        )
        
        # 強調色按鈕
        self.style.configure(
            "Accent.TButton", 
            background=self.colors["accent"], 
            foreground="#ffffff", 
            font=("Segoe UI", 9, "bold"),
            padding=(8, 4),
            relief="flat"
        )
        self.style.map("Accent.TButton",
            background=[('pressed', '#008b91'), ('active', self.colors["accent_hover"]), ('disabled', '#2a2e39')],
            foreground=[('disabled', '#6b7280')]
        )

    def setup_ui(self):
        # 頂部導航列 (包含語言切換)
        top_bar = tk.Frame(self.window, bg=self.colors["bg_card"], height=42)
        top_bar.pack(side=tk.TOP, fill=tk.X)
        
        self.lbl_app_title = tk.Label(top_bar, text=self.t("title"), bg=self.colors["bg_card"], fg="#00f7ff", font=("Segoe UI", 12, "bold"))
        self.lbl_app_title.pack(side=tk.LEFT, padx=14, pady=6)
        
        # 語言下拉選單
        lang_box = tk.Frame(top_bar, bg=self.colors["bg_card"])
        lang_box.pack(side=tk.RIGHT, padx=14)
        
        tk.Label(lang_box, text="🌐 Language:", bg=self.colors["bg_card"], fg=self.colors["text_secondary"], font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=4)
        
        self.lang_map = {
            "繁體中文": "zh_TW",
            "简体中文": "zh_CN",
            "English": "en",
            "日本語": "ja",
            "Español": "es"
        }
        self.combo_lang = ttk.Combobox(lang_box, textvariable=self.var_lang_choice, values=list(self.lang_map.keys()), state="readonly", width=10)
        self.combo_lang.pack(side=tk.LEFT)
        self.combo_lang.bind("<<ComboboxSelected>>", self.on_language_selected)
        
        # 主工作區 (左右分欄)
        main_workspace = tk.Frame(self.window, bg=self.colors["bg_main"])
        main_workspace.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        
        # 左側控制與影像面板
        self.left_frame = tk.Frame(main_workspace, bg=self.colors["bg_main"], width=660)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 6), pady=4)
        
        # 操作按鈕列
        self.btn_frame = tk.Frame(self.left_frame, bg=self.colors["bg_main"])
        self.btn_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))
        
        self.btn_open = ttk.Button(self.btn_frame, text=self.t("open_video"), command=self.open_video, style="Modern.TButton")
        self.btn_open.pack(side=tk.LEFT, padx=2)
        
        self.btn_scale = ttk.Button(self.btn_frame, text=self.t("set_scale"), command=self.start_calibration, state=tk.DISABLED, style="Modern.TButton")
        self.btn_scale.pack(side=tk.LEFT, padx=2)
        
        self.btn_select = ttk.Button(self.btn_frame, text=self.t("select_target"), command=self.start_selection, state=tk.DISABLED, style="Modern.TButton")
        self.btn_select.pack(side=tk.LEFT, padx=2)
        
        self.btn_start = ttk.Button(self.btn_frame, text=self.t("track_play"), command=self.start_tracking, state=tk.DISABLED, style="Accent.TButton")
        self.btn_start.pack(side=tk.LEFT, padx=2)
        
        self.btn_fast_start = ttk.Button(self.btn_frame, text=self.t("fast_track"), command=self.start_fast_tracking, state=tk.DISABLED, style="Modern.TButton")
        self.btn_fast_start.pack(side=tk.LEFT, padx=2)
        
        self.btn_export = ttk.Button(self.btn_frame, text=self.t("export_csv"), command=self.open_export_dialog, state=tk.DISABLED, style="Modern.TButton")
        self.btn_export.pack(side=tk.RIGHT, padx=2)

        # 演算法與影像設定框
        self.frame_algo = ttk.LabelFrame(self.left_frame, text=self.t("algo_settings"), padding=6)
        self.frame_algo.pack(side=tk.TOP, fill=tk.X, pady=3)
        
        algo_row = tk.Frame(self.frame_algo, bg=self.colors["bg_card"])
        algo_row.pack(fill=tk.X, pady=1)
        
        self.lbl_algo = tk.Label(algo_row, text=self.t("algo_label"), bg=self.colors["bg_card"], fg=self.colors["text_primary"])
        self.lbl_algo.pack(side=tk.LEFT)
        self.combo_algo = ttk.Combobox(algo_row, textvariable=self.var_tracker_type, values=["CSRT", "KCF", "MedianFlow", "MOSSE"], width=10, state="readonly")
        self.combo_algo.pack(side=tk.LEFT, padx=4)
        
        self.lbl_prep = tk.Label(algo_row, text=self.t("enhance_label"), bg=self.colors["bg_card"], fg=self.colors["text_primary"])
        self.lbl_prep.pack(side=tk.LEFT, padx=(10, 0))
        self.combo_prep = ttk.Combobox(algo_row, textvariable=self.var_preprocess_mode, values=self.t("prep_options"), width=18, state="readonly")
        self.combo_prep.current(0)
        self.combo_prep.pack(side=tk.LEFT, padx=4)
        
        self.chk_kalman = ttk.Checkbutton(self.frame_algo, text=self.t("kalman_enable"), variable=self.var_use_kalman)
        self.chk_kalman.pack(anchor=tk.W, pady=(3, 1))

        # 速度與取樣控制框
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
        
        self.chk_skip = ttk.Checkbutton(
            skip_row, text=self.t("skip_frames"), variable=self.var_enable_skipping, 
            command=self.update_skip_prediction
        )
        self.chk_skip.pack(side=tk.LEFT, padx=4)
        
        self.lbl_prediction = tk.Label(
            skip_row, text=self.t("predict_single"), 
            bg=self.colors["bg_card"], fg=self.colors["accent"], font=("Segoe UI", 8, "bold")
        )
        self.lbl_prediction.pack(side=tk.LEFT, padx=6)

        # 影像 Canvas
        self.canvas = tk.Canvas(self.left_frame, width=640, height=480, bg=self.colors["canvas_bg"], highlightthickness=1, highlightbackground=self.colors["bg_card_border"])
        self.canvas.pack(side=tk.TOP, pady=4)
        
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
        
        # 標尺資訊
        self.lbl_scale_info = tk.Label(self.left_frame, text=self.t("scale_not_set"), bg=self.colors["bg_main"], fg="#38bdf8", font=("Segoe UI", 9, "bold"))
        self.lbl_scale_info.pack(side=tk.TOP, anchor=tk.W, pady=1)

        # 底部狀態列
        bottom_bar = tk.Frame(self.left_frame, bg=self.colors["bg_main"])
        bottom_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=2)
        
        self.btn_info = ttk.Button(bottom_bar, text=self.t("info_btn"), command=self.show_about_info, style="Modern.TButton")
        self.btn_info.pack(side=tk.LEFT, padx=2)
        
        self.lbl_status = tk.Label(bottom_bar, text=self.t("status_ready"), bg=self.colors["bg_main"], fg="#60a5fa", font=("Segoe UI", 9))
        self.lbl_status.pack(side=tk.LEFT, padx=8)

        # 右側面板：圖表與自選勾選列
        self.right_frame = tk.Frame(main_workspace, bg=self.colors["bg_main"])
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.checkbox_panel = ttk.LabelFrame(self.right_frame, text=self.t("metrics_panel"), padding=6)
        self.checkbox_panel.pack(side=tk.TOP, fill=tk.X, padx=4, pady=2)
        
        control_subrow = tk.Frame(self.checkbox_panel, bg=self.colors["bg_card"])
        control_subrow.pack(side=tk.TOP, fill=tk.X, pady=1)
        
        self.chk_smooth = ttk.Checkbutton(
            control_subrow, text=self.t("smooth_chart"), 
            variable=self.var_smooth_chart, command=self.update_all_charts
        )
        self.chk_smooth.pack(side=tk.LEFT, padx=4)
        
        self.metrics_grid = tk.Frame(self.checkbox_panel, bg=self.colors["bg_card"])
        self.metrics_grid.pack(side=tk.TOP, fill=tk.X, pady=(4, 0))
        
        default_checked = ["Coord_X", "Coord_Y", "Speed", "Accel_Mag", "Path_Length"]
        col_idx, row_idx = 0, 0
        for key in self.metric_colors.keys():
            var = tk.BooleanVar(value=(key in default_checked))
            self.metric_vars[key] = var
            chk = ttk.Checkbutton(
                self.metrics_grid, text=key, variable=var,
                command=self.rebuild_chart_container
            )
            chk.grid(row=row_idx, column=col_idx, sticky="w", padx=4, pady=1)
            self.metric_checkboxes[key] = chk
            col_idx += 1
            if col_idx >= 4:
                col_idx = 0
                row_idx += 1

        # 圖表滾動容器
        chart_scroll_frame = tk.Frame(self.right_frame, bg=self.colors["bg_main"])
        chart_scroll_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=4, pady=4)
        
        self.plot_canvas = tk.Canvas(chart_scroll_frame, bg="#1a1d24", highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(chart_scroll_frame, orient="vertical", command=self.plot_canvas.yview)
        self.scrollable_inner_frame = tk.Frame(self.plot_canvas, bg="#1a1d24")
        
        self.scrollable_inner_frame.bind(
            "<Configure>",
            lambda e: self.plot_canvas.configure(scrollregion=self.plot_canvas.bbox("all"))
        )
        self.plot_canvas.create_window((0, 0), window=self.scrollable_inner_frame, anchor="nw")
        self.plot_canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.plot_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.plot_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def on_language_selected(self, event=None):
        selected_name = self.var_lang_choice.get()
        lang_code = self.lang_map.get(selected_name, "zh_TW")
        self.change_language(lang_code)

    def change_language(self, lang_code, force=False):
        if self.current_lang == lang_code and not force:
            return
        self.current_lang = lang_code
        apply_adaptive_matplotlib_font(lang_code)
        
        # 更新文字
        self.window.title(self.t("title"))
        self.lbl_app_title.config(text=self.t("title"))
        self.btn_open.config(text=self.t("open_video"))
        self.btn_scale.config(text=self.t("set_scale"))
        self.btn_select.config(text=self.t("select_target"))
        self.btn_start.config(text=self.t("track_play"))
        self.btn_fast_start.config(text=self.t("fast_track"))
        self.btn_export.config(text=self.t("export_csv"))
        self.btn_info.config(text=self.t("info_btn"))
        self.btn_reset_speed.config(text=self.t("reset_speed"))
        
        self.frame_algo.config(text=self.t("algo_settings"))
        self.lbl_algo.config(text=self.t("algo_label"))
        self.lbl_prep.config(text=self.t("enhance_label"))
        self.chk_kalman.config(text=self.t("kalman_enable"))
        
        curr_prep_idx = self.combo_prep.current()
        self.combo_prep.config(values=self.t("prep_options"))
        if curr_prep_idx >= 0:
            self.combo_prep.current(curr_prep_idx)
            
        self.frame_speed.config(text=self.t("play_sampling"))
        self.chk_skip.config(text=self.t("skip_frames"))
        self.checkbox_panel.config(text=self.t("metrics_panel"))
        self.chk_smooth.config(text=self.t("smooth_chart"))
        
        # 更新物理量勾選名稱
        metric_dict = self.t("metrics")
        for k, chk in self.metric_checkboxes.items():
            if k in metric_dict:
                chk.config(text=metric_dict[k][0])
                
        # 更新比例尺與提示狀態
        if self.pixel_per_unit:
            self.lbl_scale_info.config(text=self.t("scale_set").format(unit=self.unit_name, scale=self.pixel_per_unit))
        else:
            self.lbl_scale_info.config(text=self.t("scale_not_set"))
            
        self.update_skip_prediction()
        self.rebuild_chart_container()

    def show_about_info(self):
        info_text = (
            f"{self.t('title')}\n\n"
            "Architecture: OpenCV (CSRT/KCF/MedianFlow/MOSSE) + Kalman Filter\n"
            "Image Processing: Adaptive CLAHE + Gaussian Denoising\n"
            "Kinematics: Finite Difference Velocity/Acceleration/Jerk Calculation\n"
            "Languages: 繁體中文, 简体中文, English, 日本語, Español\n"
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
        if prep_idx == 2:  # Raw
            return frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prep_idx == 0:  # CLAHE
            enhanced = self.clahe.apply(gray)
            return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        elif prep_idx == 1:  # Denoise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            return cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
        return frame

    def open_video(self):
        if self.cap is not None:
            self.cap.release()
            
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv *.wmv")])
        if not self.video_path:
            return
        
        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0 or np.isnan(self.fps): 
            self.fps = 30.0
        
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
            self.lbl_scale_info.config(text=self.t("scale_not_set"))
            
            self.display_frame(self.current_frame)
            self.lbl_status.config(text=self.t("status_loaded").format(fps=self.fps))
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
        self.lbl_status.config(text=self.t("status_calibrating"))
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        _, self.current_frame = self.cap.read()
        self.display_frame(self.current_frame)

    def on_mouse_press(self, event):
        if self.calibrating:
            orig_h, orig_w, _ = self.current_frame.shape
            real_x = int(event.x * (orig_w / 640))
            real_y = int(event.y * (orig_h / 480))
            
            self.scale_points.append((real_x, real_y))
            self.canvas.create_oval(event.x-4, event.y-4, event.x+4, event.y+4, fill="#facc15", tags="scale_dot")
            
            if len(self.scale_points) == 2:
                self.calibrating = False
                p1, p2 = self.scale_points[0], self.scale_points[1]
                pixel_dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                
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
        self.lbl_status.config(text=self.t("status_locked"))
        self.btn_start.config(state=tk.NORMAL)
        self.btn_fast_start.config(state=tk.NORMAL)

    def create_tracker_instance(self):
        algo = self.var_tracker_type.get()
        creator = {
            "CSRT": lambda: cv2.legacy.TrackerCSRT_create() if hasattr(cv2, 'legacy') else cv2.TrackerCSRT_create(),
            "KCF": lambda: cv2.legacy.TrackerKCF_create() if hasattr(cv2, 'legacy') else cv2.TrackerKCF_create(),
            "MedianFlow": lambda: cv2.legacy.TrackerMedianFlow_create() if hasattr(cv2, 'legacy') else cv2.TrackerMedianFlow_create(),
            "MOSSE": lambda: cv2.legacy.TrackerMOSSE_create() if hasattr(cv2, 'legacy') else cv2.TrackerMOSSE_create()
        }
        try:
            return creator.get(algo, creator["CSRT"])()
        except Exception:
            return cv2.TrackerCSRT_create()

    def init_clean_tracker(self):
        self.tracker = self.create_tracker_instance()
        base_frame = self.first_frame if self.first_frame is not None else self.current_frame
        proc_frame = self.preprocess_frame(base_frame)
        self.tracker.init(proc_frame, self.bbox)
        
        cx = self.bbox[0] + self.bbox[2] / 2.0
        cy = self.bbox[1] + self.bbox[3] / 2.0
        self.kalman.init(cx, cy)

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
        self.lbl_status.config(text=self.t("select_target_prompt"))
        self.update_all_charts()

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
            if step_px > 2.0:
                tot_path = prev.get("Path_Length", 0.0) + np.sqrt(dx**2 + dy**2)
            else:
                tot_path = prev.get("Path_Length", 0.0)
            
            vel_x = dx / dt
            vel_y = dy / dt
            speed = np.sqrt(vel_x**2 + vel_y**2)
            
            raw_accel_x = (vel_x - prev["Vel_X"]) / dt
            raw_accel_y = (vel_y - prev["Vel_Y"]) / dt
            accel_x = 0.7 * raw_accel_x + 0.3 * prev.get("Accel_X", 0.0)
            accel_y = 0.7 * raw_accel_y + 0.3 * prev.get("Accel_Y", 0.0)
            accel_mag = np.sqrt(accel_x**2 + accel_y**2)
            
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
        self.lbl_status.config(text=self.t("status_fast_running"))
        self.window.update()
        
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            
            self.frame_count += 1
            proc_frame = self.preprocess_frame(frame)
            success, bbox = self.tracker.update(proc_frame)
            
            cx, cy = None, None
            if success:
                self.lost_frames_counter = 0
                x, y, w, h = [int(v) for v in bbox]
                raw_cx, raw_cy = x + w/2.0, y + h/2.0
                if self.var_use_kalman.get():
                    self.kalman.predict()
                    cx, cy = self.kalman.correct(raw_cx, raw_cy)
                else:
                    cx, cy = raw_cx, raw_cy
            else:
                self.lost_frames_counter += 1
                if self.var_use_kalman.get() and self.lost_frames_counter < 10:
                    cx, cy = self.kalman.predict()
                elif self.lost_frames_counter >= self.lost_frames_limit:
                    break

            if cx is not None and cy is not None:
                row = self.process_kinematics(cx, cy, self.frame_count)
                self.frame_data.append(row)
        
        self.update_all_charts()
        self.end_tracking(self.t("status_tracking_done"))

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
            self.end_tracking(self.t("status_tracking_done"))
            return
            
        self.current_frame = frame.copy()
        proc_frame = self.preprocess_frame(frame)
        success, bbox = self.tracker.update(proc_frame)
        
        cx, cy = None, None
        status_color = (0, 255, 0)
        
        if success:
            self.lost_frames_counter = 0
            x, y, w, h = [int(v) for v in bbox]
            raw_cx, raw_cy = x + w/2.0, y + h/2.0
            
            if self.var_use_kalman.get():
                self.kalman.predict()
                cx, cy = self.kalman.correct(raw_cx, raw_cy)
            else:
                cx, cy = raw_cx, raw_cy
                
            cv2.rectangle(frame, (x, y), (x + w, y + h), status_color, 2)
        else:
            self.lost_frames_counter += 1
            if self.var_use_kalman.get() and self.lost_frames_counter < 15:
                cx, cy = self.kalman.predict()
                status_color = (0, 165, 255)
                cv2.circle(frame, (int(cx), int(cy)), 10, status_color, 2)
            elif self.lost_frames_counter >= self.lost_frames_limit:
                self.update_all_charts()
                self.end_tracking(self.t("status_lost"))
                return

        if cx is not None and cy is not None:
            row = self.process_kinematics(cx, cy, self.frame_count)
            self.frame_data.append(row)
            
            if len(self.frame_data) > 1:
                trail_len = min(40, len(self.frame_data))
                for i in range(len(self.frame_data) - trail_len + 1, len(self.frame_data)):
                    pt1 = (int(self.frame_data[i-1]["Raw_X"]), int(self.frame_data[i-1]["Raw_Y"]))
                    pt2 = (int(self.frame_data[i]["Raw_X"]), int(self.frame_data[i]["Raw_Y"]))
                    cv2.line(frame, pt1, pt2, (0, 220, 255), 2, cv2.LINE_AA)
            
            cv2.circle(frame, (int(cx), int(cy)), 4, (255, 50, 50), -1)
            cv2.putText(frame, f"v: {row['Speed']:.1f} {self.unit_name}/s | |a|: {row['Accel_Mag']:.1f}", 
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
        if not selected_metrics:
            return
            
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
            
            btn_view = ttk.Button(
                card_frame, text=self.t("view_data"), 
                style="Modern.TButton",
                command=lambda k=key: self.open_single_metric_viewer(k)
            )
            btn_view.pack(side=tk.RIGHT, padx=8, pady=10)
            
            self.chart_cards[key] = {
                "fig": fig, "ax": ax, "canvas": canvas,
                "color": color, "label": label_name, "unit": unit_pattern
            }
            
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
        summary_text = self.t("summary_fmt").format(
            n=len(df), max_v=val_series.max(), unit=actual_unit, mean_v=val_series.mean()
        )
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

    def open_export_dialog(self):
        if not self.frame_data:
            messagebox.showwarning(self.t("title"), self.t("no_data_warn"))
            return
            
        export_win = tk.Toplevel(self.window)
        export_win.title(self.t("export_title"))
        export_win.geometry("460x500")
        export_win.resizable(False, False)
        export_win.configure(bg=self.colors["bg_card"])
        
        tk.Label(export_win, text=self.t("export_prompt"), bg=self.colors["bg_card"], fg="#00f7ff", font=("Segoe UI", 10, "bold")).pack(pady=10)
        
        check_frame = tk.Frame(export_win, bg=self.colors["bg_card"], padx=10)
        check_frame.pack(fill=tk.BOTH, expand=True)
        
        export_vars = {}
        metric_dict = self.t("metrics")
        for key in self.metric_colors.keys():
            label_text = metric_dict.get(key, (key, ""))[0]
            initial_val = self.metric_vars.get(key, tk.BooleanVar(value=True)).get()
            var = tk.BooleanVar(value=initial_val)
            export_vars[key] = var
            chk = ttk.Checkbutton(check_frame, text=label_text, variable=var)
            chk.pack(anchor="w", pady=2)
            
        btn_action_frame = tk.Frame(export_win, bg=self.colors["bg_card"], padx=10, pady=10)
        btn_action_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        def select_all(state):
            for v in export_vars.values(): v.set(state)
            
        ttk.Button(btn_action_frame, text=self.t("select_all"), command=lambda: select_all(True), style="Modern.TButton").pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_action_frame, text=self.t("deselect_all"), command=lambda: select_all(False), style="Modern.TButton").pack(side=tk.LEFT, padx=4)
        
        def execute_export():
            chosen_metrics = [k for k, v in export_vars.items() if v.get()]
            if not chosen_metrics:
                return
                
            save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
            if not save_path: return
            
            df = pd.DataFrame(self.frame_data)
            export_cols = ["Frame", "Time"] + chosen_metrics
            df[export_cols].to_csv(save_path, index=False)
            
            export_win.destroy()
            msg = self.t("export_success").format(count=len(chosen_metrics), path=save_path)
            messagebox.showinfo(self.t("title"), msg)
            
        ttk.Button(btn_action_frame, text=self.t("confirm_export"), command=execute_export, style="Accent.TButton").pack(side=tk.RIGHT, padx=4)

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
        self.lbl_status.config(text=message)

if __name__ == "__main__":
    root = tk.Tk()
    app = UniversalPhysicsTrackerApp(root)
    root.mainloop()