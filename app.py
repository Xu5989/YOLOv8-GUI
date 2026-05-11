import cv2
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
from ultralytics import YOLO

# --- 全局 UI 主题设置 ---
ctk.set_appearance_mode("System")  # 跟随系统主题 (深色/浅色)
ctk.set_default_color_theme("blue")  # 组件主色调


class EmotionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # 1. 窗口基础设置bes
        self.title("🚀 YOLOv8 面部表情智能监测系统")
        self.geometry("1200x750")
        self.minsize(900, 600)

        # 2. 加载模型
        self.model_path = 'best.pt'
        try:
            self.model = YOLO(self.model_path)
            print("✅ 模型加载成功！")
        except Exception as e:
            messagebox.showerror("模型加载失败", f"找不到模型文件: {self.model_path}\n{e}")
            self.destroy()
            return

        self.vid = None
        self.is_camera_on = False
        self.is_video_playing = False  # 新增：标记视频是否正在播放
        self.current_image_rgb = None  # 用于保存当前屏幕上的画面，以便“暂存”

        # 3. 核心布局：1x2 网格 (左侧控制台，右侧主屏幕)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # ==================== 左侧：控制台面板 ====================
        self.sidebar = ctk.CTkFrame(self, width=260, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(7, weight=1)  # 修改：让画廊区域自动拉伸 (向下移了一行)

        # 标题栏
        self.logo_label = ctk.CTkLabel(self.sidebar, text="AI 监测控制台", font=ctk.CTkFont(size=22, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(30, 20))

        # 操作按钮
        self.btn_upload = ctk.CTkButton(self.sidebar, text="📂 上传图片检测", height=40, command=self.upload_image)
        self.btn_upload.grid(row=1, column=0, padx=20, pady=10)

        # 新增：上传视频按钮
        self.btn_upload_video = ctk.CTkButton(self.sidebar, text="🎞️ 上传视频检测", height=40,
                                              command=self.upload_video)
        self.btn_upload_video.grid(row=2, column=0, padx=20, pady=10)

        self.btn_camera = ctk.CTkButton(self.sidebar, text="📷 开启实时摄像头", height=40, command=self.start_camera)
        self.btn_camera.grid(row=3, column=0, padx=20, pady=10)

        # 修改：统一的停止按钮
        self.btn_stop = ctk.CTkButton(self.sidebar, text="🚫 停止当前画面", height=40, fg_color="gray", state="disabled",
                                      command=self.stop_media)
        self.btn_stop.grid(row=4, column=0, padx=20, pady=10)

        self.btn_snapshot = ctk.CTkButton(self.sidebar, text="📸 抓拍当前画面 (暂存)", height=40, fg_color="#2FA572",
                                          hover_color="#1F7A52", command=self.take_snapshot)
        self.btn_snapshot.grid(row=5, column=0, padx=20, pady=(30, 10))

        # 暂存画廊 (滚动区域)
        self.gallery_frame = ctk.CTkScrollableFrame(self.sidebar, label_text="🎞️ 暂存记录 ",
                                                    label_font=ctk.CTkFont(weight="bold"))
        self.gallery_frame.grid(row=7, column=0, padx=10, pady=(10, 20), sticky="nsew")

        # ==================== 右侧：主显示区域 ====================
        self.main_frame = ctk.CTkFrame(self, corner_radius=15)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        self.video_label = ctk.CTkLabel(self.main_frame, text="等待画面输入...", font=ctk.CTkFont(size=24),
                                        text_color="gray")
        self.video_label.grid(row=0, column=0)

        # 窗口关闭协议
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    # --- 功能函数区 ---
    def upload_image(self):
        """上传图片"""
        self.stop_media()  # 确保关闭其他视频流
        file_path = filedialog.askopenfilename(title="选择图片", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            frame = cv2.imread(file_path)
            if frame is not None:
                self.detect_and_display(frame)

    def upload_video(self):
        """上传视频并播放检测"""
        self.stop_media()
        file_path = filedialog.askopenfilename(title="选择视频", filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
        if file_path:
            self.vid = cv2.VideoCapture(file_path)
            if not self.vid.isOpened():
                messagebox.showerror("错误", "无法打开视频文件！")
                return

            self.is_video_playing = True
            self.btn_upload_video.configure(state="disabled", fg_color="gray")
            self.btn_camera.configure(state="normal", fg_color="#1f538d")
            self.btn_stop.configure(state="normal", fg_color="#1f538d")
            self.update_video()

    def update_video(self):
        """视频循环刷新"""
        if self.is_video_playing:
            ret, frame = self.vid.read()
            if ret:
                self.detect_and_display(frame)
                # 33ms 延迟，大约对应 30 FPS 的播放速度
                self.after(10, self.update_video)
            else:
                # 视频播放完毕
                self.stop_media()
                self.video_label.configure(image=None, text="视频播放结束")

    def start_camera(self):
        """开启摄像头"""
        self.stop_media()
        if not self.is_camera_on:
            self.vid = cv2.VideoCapture(0)
            if not self.vid.isOpened():
                messagebox.showerror("错误", "无法打开摄像头！请检查权限。")
                return

            self.is_camera_on = True
            self.btn_camera.configure(state="disabled", fg_color="gray")
            self.btn_upload_video.configure(state="normal", fg_color="#1f538d")
            self.btn_stop.configure(state="normal", fg_color="#1f538d")
            self.update_camera()

    def update_camera(self):
        """摄像头循环刷新"""
        if self.is_camera_on:
            ret, frame = self.vid.read()
            if ret:
                frame = cv2.flip(frame, 1)  # 镜像翻转
                self.detect_and_display(frame)
            # 摄像头刷新延迟
            self.after(15, self.update_camera)

    def stop_media(self):
        """统一关闭媒体流 (摄像头或视频)"""
        self.is_camera_on = False
        self.is_video_playing = False
        if self.vid:
            self.vid.release()
            self.vid = None

        self.video_label.configure(image=None, text="画面已关闭")
        self.btn_camera.configure(state="normal", fg_color="#1f538d")
        self.btn_upload_video.configure(state="normal", fg_color="#1f538d")
        self.btn_stop.configure(state="disabled", fg_color="gray")

    def detect_and_display(self, frame):
        """YOLO推理并显示到主屏幕"""
        # 模型推理
        results = self.model.predict(source=frame, conf=0.425, verbose=False)
        annotated_frame = results[0].plot()

        # OpenCV(BGR) 转 PIL(RGB)
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(annotated_frame)
        self.current_image_rgb = pil_img  # 保存当前帧，用于抓拍功能

        # 动态计算缩放比例，保持长宽比填满主屏幕区域
        img_w, img_h = pil_img.size
        ratio = min(900 / img_w, 700 / img_h)
        new_w, new_h = int(img_w * ratio), int(img_h * ratio)

        # 使用 CTkImage 显示，自带抗锯齿和主题适配
        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(new_w, new_h))
        self.video_label.configure(image=ctk_img, text="")
        self.video_label.image = ctk_img

    def take_snapshot(self):
        """抓拍当前画面并暂存到左侧画廊"""
        if self.current_image_rgb is not None:
            # 创建一个缩略图，长宽比例缩小
            img_w, img_h = self.current_image_rgb.size
            thumb_ratio = 200 / img_w
            thumb_w, thumb_h = int(img_w * thumb_ratio), int(img_h * thumb_ratio)

            thumb_img = ctk.CTkImage(light_image=self.current_image_rgb, dark_image=self.current_image_rgb,
                                     size=(thumb_w, thumb_h))

            # 将缩略图作为一个 Label 添加到暂存画廊滚动条中
            snapshot_label = ctk.CTkLabel(self.gallery_frame, image=thumb_img, text="")
            snapshot_label.pack(pady=5)

            # 按钮给出视觉反馈
            self.btn_snapshot.configure(text="✅ 抓拍成功！")
            self.after(1000, lambda: self.btn_snapshot.configure(text="📸 抓拍当前画面 (暂存)"))

    def on_closing(self):
        self.stop_media()
        self.destroy()


if __name__ == '__main__':
    app = EmotionApp()
    app.mainloop()