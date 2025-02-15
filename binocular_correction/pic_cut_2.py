import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os

class ImageCropperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("圖片剪裁工具")
        self.root.geometry("1000x800")
        
        # 建立主要框架
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 變數初始化
        self.folder_path = tk.StringVar()
        self.current_image = None
        self.preview_photo = None
        self.original_image = None
        self.current_file_path = None
        
        # 記住裁切範圍的變數
        self.saved_crop_coords = None
        self.use_saved_coords = tk.BooleanVar(value=False)
        
        # 剪裁區域座標
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None
        self.dragging = False
        
        # 控制點變數
        self.control_points = []
        self.active_point = None
        self.point_size = 6
        self.control_point_tags = ["top_left", "top_right", "bottom_left", "bottom_right"]
        
        # 建立UI元件
        self.create_widgets()
        
        # 檔案列表
        self.file_list = []
        self.current_file_index = -1
        
    def create_widgets(self):
        # 選擇資料夾區域
        folder_frame = ttk.Frame(self.main_frame)
        folder_frame.grid(row=0, column=0, columnspan=3, sticky=tk.W+tk.E, pady=5)
        
        ttk.Label(folder_frame, text="選擇資料夾:").pack(side=tk.LEFT)
        ttk.Entry(folder_frame, textvariable=self.folder_path, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(folder_frame, text="瀏覽", command=self.browse_folder).pack(side=tk.LEFT)
        
        # 預覽區域
        self.canvas = tk.Canvas(self.main_frame, width=800, height=600, bg='gray')
        self.canvas.grid(row=1, column=0, columnspan=3, pady=10)
        
        # 綁定滑鼠事件
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Motion>", self.update_cursor)
        
        # 按鈕區域
        button_frame = ttk.Frame(self.main_frame)
        button_frame.grid(row=2, column=0, columnspan=3, pady=5)
        
        ttk.Button(button_frame, text="上一張", command=self.prev_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="下一張", command=self.next_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="重設選擇", command=self.reset_selection).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="剪裁", command=self.crop_current_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="批次剪裁", command=self.batch_crop).pack(side=tk.LEFT, padx=5)
        
        ttk.Checkbutton(button_frame, text="使用已儲存的裁切範圍", 
                       variable=self.use_saved_coords).pack(side=tk.LEFT, padx=5)
        
        # 狀態顯示
        self.status_label = ttk.Label(self.main_frame, text="")
        self.status_label.grid(row=3, column=0, columnspan=3, pady=5)
        
    def on_press(self, event):
        x, y = event.x, event.y
        
        # 檢查是否點擊到控制點
        for i, point in enumerate(self.control_points):
            px, py = point
            if abs(x - px) <= self.point_size and abs(y - py) <= self.point_size:
                self.active_point = i
                return
        
        # 如果沒有點擊到控制點，就開始新的選取
        self.start_x = x
        self.start_y = y
        self.end_x = x
        self.end_y = y
        self.dragging = True
        self.canvas.delete("crop_box")
        self.canvas.delete("control_point")
        self.control_points = []

    def on_drag(self, event):
        if self.active_point is not None:
            # 調整控制點
            x, y = event.x, event.y
            if self.active_point == 0:  # 左上
                self.start_x, self.start_y = x, y
            elif self.active_point == 1:  # 右上
                self.end_x, self.start_y = x, y
            elif self.active_point == 2:  # 左下
                self.start_x, self.end_y = x, y
            elif self.active_point == 3:  # 右下
                self.end_x, self.end_y = x, y
        elif self.dragging:
            # 新選取
            self.end_x = event.x
            self.end_y = event.y
        
        self.update_selection()

    def on_release(self, event):
        self.dragging = False
        self.active_point = None
        self.update_selection()

    def update_cursor(self, event):
        x, y = event.x, event.y
        
        # 檢查滑鼠是否在控制點上
        for point in self.control_points:
            px, py = point
            if abs(x - px) <= self.point_size and abs(y - py) <= self.point_size:
                self.canvas.config(cursor="crosshair")
                return
        
        self.canvas.config(cursor="")

    def update_selection(self):
        self.canvas.delete("crop_box")
        self.canvas.delete("control_point")
        
        if None not in (self.start_x, self.start_y, self.end_x, self.end_y):
            # 繪製選取框
            self.canvas.create_rectangle(
                self.start_x, self.start_y, self.end_x, self.end_y,
                outline="red", width=2, tags="crop_box"
            )
            
            # 更新控制點位置
            self.control_points = [
                (self.start_x, self.start_y),  # 左上
                (self.end_x, self.start_y),    # 右上
                (self.start_x, self.end_y),    # 左下
                (self.end_x, self.end_y)       # 右下
            ]
            
            # 繪製控制點
            for (x, y), tag in zip(self.control_points, self.control_point_tags):
                self.canvas.create_oval(
                    x - self.point_size, y - self.point_size,
                    x + self.point_size, y + self.point_size,
                    fill="white", outline="red",
                    tags=("control_point", tag)
                )
    
    def browse_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.folder_path.set(folder_selected)
            self.load_file_list()
            self.current_file_index = -1
            self.next_image()
    
    def load_file_list(self):
        folder = self.folder_path.get()
        if not folder:
            return
            
        self.file_list = []
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP')
        for filename in os.listdir(folder):
            if filename.lower().endswith(image_extensions):
                self.file_list.append(filename)
        self.file_list.sort()
        
    def load_image(self, file_path):
        try:
            image = Image.open(file_path)
            self.original_image = image
            
            # 計算縮放比例，讓圖片適應 canvas
            ratio = min(800/image.width, 600/image.height)
            new_width = int(image.width * ratio)
            new_height = int(image.height * ratio)
            
            # 調整圖片大小以適應預覽窗口
            display_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.current_image = display_image
            self.preview_photo = ImageTk.PhotoImage(display_image)
            
            # 更新 canvas
            self.canvas.delete("all")
            self.canvas.create_image(400, 300, anchor=tk.CENTER, image=self.preview_photo)
            
            # 重設剪裁區域
            self.reset_selection()
            
            self.status_label.configure(text=f"目前檔案: {os.path.basename(file_path)}")
            
        except Exception as e:
            messagebox.showerror("錯誤", f"無法載入圖片: {str(e)}")
    
    def reset_selection(self):
        self.canvas.delete("crop_box")
        self.canvas.delete("control_point")
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None
        self.control_points = []
        if not self.use_saved_coords.get():
            self.saved_crop_coords = None
    
    def batch_crop(self):
        if self.saved_crop_coords is None:
            messagebox.showwarning("警告", "請先在第一張圖片上選擇裁切範圍")
            return
            
        if not self.file_list:
            return
            
        try:
            total = len(self.file_list)
            processed = 0
            
            for filename in self.file_list:
                file_path = os.path.join(self.folder_path.get(), filename)
                self.current_file_path = file_path
                self.load_image(file_path)
                
                # 強制使用已儲存的裁切範圍
                self.use_saved_coords.set(True)
                self.crop_current_image()
                
                processed += 1
                self.status_label.configure(text=f"批次處理中: {processed}/{total}")
                self.root.update()
            
            messagebox.showinfo("完成", f"已完成批次處理 {processed} 張圖片")
            
        except Exception as e:
            messagebox.showerror("錯誤", str(e))
    
    def crop_current_image(self):
        if self.current_file_path is None:
            return
            
        if None in (self.start_x, self.start_y, self.end_x, self.end_y):
            if not self.use_saved_coords.get() or self.saved_crop_coords is None:
                messagebox.showwarning("警告", "請先選擇剪裁區域")
                return
            else:
                # 使用儲存的裁切範圍
                ratio_x1, ratio_y1, ratio_x2, ratio_y2 = self.saved_crop_coords
                display_width = self.current_image.width
                display_height = self.current_image.height
                canvas_x = (800 - display_width) / 2
                canvas_y = (600 - display_height) / 2
                self.start_x = ratio_x1 * display_width + canvas_x
                self.start_y = ratio_y1 * display_height + canvas_y
                self.end_x = ratio_x2 * display_width + canvas_x
                self.end_y = ratio_y2 * display_height + canvas_y
            
        try:
            # 取得選擇區域座標
            x1, x2 = sorted([self.start_x, self.end_x])
            y1, y2 = sorted([self.start_y, self.end_y])
            
            # 計算原始圖片對應的剪裁區域
            display_width = self.current_image.width
            display_height = self.current_image.height
            original_width = self.original_image.width
            original_height = self.original_image.height
            
            # 計算縮放比例
            width_ratio = original_width / display_width
            height_ratio = original_height / display_height
            
            # 轉換座標到原始圖片的尺寸
            crop_x1 = int((x1 - (800 - display_width) / 2) * width_ratio)
            crop_y1 = int((y1 - (600 - display_height) / 2) * height_ratio)
            crop_x2 = int((x2 - (800 - display_width) / 2) * width_ratio)
            crop_y2 = int((y2 - (600 - display_height) / 2) * height_ratio)
            
            # 確保座標在有效範圍內
            crop_x1 = max(0, min(crop_x1, original_width))
            crop_y1 = max(0, min(crop_y1, original_height))
            crop_x2 = max(0, min(crop_x2, original_width))
            crop_y2 = max(0, min(crop_y2, original_height))
            
            # 執行剪裁
            cropped_img = self.original_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            cropped_img.save(self.current_file_path, quality=95)
            
            # 如果是第一次裁切，儲存裁切範圍
            if self.saved_crop_coords is None:
                display_width = self.current_image.width
                display_height = self.current_image.height
                canvas_x = (800 - display_width) / 2
                canvas_y = (600 - display_height) / 2
                
                # 計算相對位置（0-1之間的比例）
                ratio_x1 = (self.start_x - canvas_x) / display_width
                ratio_y1 = (self.start_y - canvas_y) / display_height
                ratio_x2 = (self.end_x - canvas_x) / display_width
                ratio_y2 = (self.end_y - canvas_y) / display_height
                self.saved_crop_coords = (ratio_x1, ratio_y1, ratio_x2, ratio_y2)
            
            # 重新載入圖片
            self.load_image(self.current_file_path)
            
            messagebox.showinfo("成功", "圖片剪裁完成")
            
        except Exception as e:
            messagebox.showerror("錯誤", str(e))
    
    def next_image(self):
        if not self.file_list:
            return
            
        self.current_file_index = (self.current_file_index + 1) % len(self.file_list)
        self.current_file_path = os.path.join(
            self.folder_path.get(),
            self.file_list[self.current_file_index]
        )
        self.load_image(self.current_file_path)
    
    def prev_image(self):
        if not self.file_list:
            return
            
        self.current_file_index = (self.current_file_index - 1) % len(self.file_list)
        self.current_file_path = os.path.join(
            self.folder_path.get(),
            self.file_list[self.current_file_index]
        )
        self.load_image(self.current_file_path)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCropperApp(root)
    root.mainloop()