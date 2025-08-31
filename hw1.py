#-*-coding: UTF-8 -*-
import tkinter as tk #用來產生GUI
from tkinter import filedialog #用來開檔
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import math


photo=0 #存圖片
photoPath=''

historyPhotos = [] #存每次改過的圖片
history = -1
redoPhotos = [] #存復原後 復原前的東西

#復原
def undo():
    global historyPhotos, photo, history ,redoPhotos
    if(history<1):return #沒有能復原的就不執行
    photo = historyPhotos[history-1]
    history-=1
    redoPhotos.append(historyPhotos.pop()) #復原後復原前的東西先存起來
    print(redoPhotos)
    w, h = photo.size
    updata(w, h, photo, frame, 0)

#取消復原
def redo():
    global historyPhotos, photo, history
    if not redoPhotos:return #沒有能取消復原的(redoPhotos是空的)就不執行
    photo = redoPhotos.pop()
    historyPhotos.append(photo)
    history+=1
    w, h = photo.size
    updata(w, h, photo, frame, 0)
    


def clearWindows(windows): #清空畫面
    for widgets in windows.winfo_children():
        widgets.destroy()

def open():
    global photoPath, photo, frame, historyPhotos, history
    img_path = filedialog.askopenfilename(initialdir='./TestImages', filetypes=[("image", "*.jpg *.tif"),('tif', '*.tif'),('jpg', '*.jpg'),('png', '*.png')])  # 指定開啟檔案格式
    photoPath=img_path
    print(photoPath)
    img = Image.open(img_path)           # 取得圖片路徑且read
    photo=img
    w, h = img.size                      # 取得圖片長寬
    historyPhotos = []
    history = -1
    updata(w,h,img,frame, 1)
    

def updata(w, h, img, frame, notUndo):
    global history, historyPhotos, redoPhotos
    tk_img = ImageTk.PhotoImage(img)     # 轉換成 tk 圖片物件
    clearWindows(frame)
    canvas = tk.Canvas(frame, width=w, height=h, bg='#fff')
    canvas.delete('all')                 # 清空 Canvas 原本內容
    canvas.create_image(0, 0, anchor='nw', image=tk_img)   # 建立圖片
    canvas.tk_img = tk_img               # 修改屬性更新畫面
    canvas.pack()

    if(notUndo):
        history+=1
        historyPhotos.append(img)
        redoPhotos = [] #不是復原時的updata就沒有可以取消復原的東西

    draw(img)
    #clearWindows(frame2)
    #Hist = FigureCanvasTkAgg(draw(img), master=window)
    #Hist.draw() 

def save():
    global photoPath, photo
    photo.convert("L").save(photoPath) #以灰階存檔
    #photo.convert("RGB").save(photoPath)

def saveAs():
    global photo
    filepath = filedialog.asksaveasfilename(initialdir = "./TestImages", filetypes=[("image", "*.jpg *.tif"),('tif', '*.tif'),('jpg', '*.jpg'),('png', '*.png')])
    photo.convert("L").save(filepath)
    #photo.convert("RGB").save(filepath)


def draw(img): #畫每個灰階有幾個點的直方圖
    a=np.array(img)
    a=a.reshape(-1) #2d轉1d
    c=np.zeros(256,int)
    for i in range(0,256):
        c[i]=i
    plt.hist(a, bins=c)
    plt.show()


    
## 調對比跟亮度 ##
def linear():
    global enterA,enterB,enterWindow
    enterWindow=tk.Toplevel(window) #產生子視窗來輸入數值
    enterWindow.title('parameter')
    enterWindow.geometry('380x150')
    tk.Label(enterWindow, text='a').pack()
    enterA=tk.Entry(enterWindow)
    enterA.pack()
    tk.Label(enterWindow, text='b').pack()
    enterB=tk.Entry(enterWindow)
    enterB.pack()
    btn = tk.Button(enterWindow, text='enter', command=dolinear)     # 建立 Button 按鈕
    btn.pack() 

def dolinear():
    global enterA,enterB,enterWindow,photo,frame
    a=enterA.get() #取得輸入框的資料
    b=enterB.get()
    #print(a,b)
    img = np.array(photo) #圖片轉陣列

    img = img*float(a)+float(b)

    img = np.clip(img,0,255).astype('uint8') #clip(img,0,255) 超過255的當255小於0地當0 #.astype('uint8') 取整數
    photo = Image.fromarray(img)
    w, h = photo.size
    enterWindow.destroy()
    updata(w, h, photo, frame, 1)
    #print(photo)
    

def exponentail():
    global enterA,enterB,enterWindow
    enterWindow=tk.Toplevel(window) #產生子視窗來輸入數值
    enterWindow.title('parameter')
    enterWindow.geometry('380x150')
    tk.Label(enterWindow, text='a').pack()
    enterA=tk.Entry(enterWindow)
    enterA.pack()
    tk.Label(enterWindow, text='b').pack()
    enterB=tk.Entry(enterWindow)
    enterB.pack()
    btn = tk.Button(enterWindow, text='enter', command=doexp)     # 建立 Button 按鈕
    btn.pack() 

def doexp():
    global enterA,enterB,enterWindow,photo,frame
    a=enterA.get() #取得輸入框的資料
    b=enterB.get()
    print(a,b)
    img = np.array(photo) #圖片轉陣列

    img = np.exp(img*float(a)+float(b))

    img = np.clip(img,0,255).astype('uint8') #clip(img,0,255) 超過255的當255小於0地當0 #.astype('uint8') 取整數
    photo = Image.fromarray(img)
    w, h = photo.size
    enterWindow.destroy()
    updata(w, h, photo, frame, 1)
    #print(photo)
    

def logarithmical():
    global enterA,enterB,enterWindow
    enterWindow=tk.Toplevel(window) #產生子視窗來輸入數值
    enterWindow.title('parameter')
    enterWindow.geometry('380x150')
    tk.Label(enterWindow, text='a').pack()
    enterA=tk.Entry(enterWindow)
    enterA.pack()
    tk.Label(enterWindow, text='b (b>1)').pack()
    enterB=tk.Entry(enterWindow)
    enterB.pack()
    btn = tk.Button(enterWindow, text='enter', command=dolog)     # 建立 Button 按鈕
    btn.pack() 

def dolog():
    global enterA,enterB,enterWindow,photo,frame
    a=enterA.get() #取得輸入框的資料
    b=enterB.get()
    print(a,b)
    if(float(b)<=1): #輸入的數值不對就結束
        enterWindow.destroy()
        return
    img = np.array(photo) #圖片轉陣列

    img = np.log(img*float(a)+float(b))

    img = np.clip(img,0,255).astype('uint8') #clip(img,0,255) 超過255的當255小於0地當0 #.astype('uint8') 取整數
    photo = Image.fromarray(img)
    w, h = photo.size
    enterWindow.destroy()
    updata(w, h, photo, frame, 1)
    #print(photo)
    


## 調大小 ##
def resize():
    global enterA,enterWindow
    enterWindow=tk.Toplevel(window) #產生子視窗來輸入數值
    enterWindow.title('parameter')
    enterWindow.geometry('380x150')
    tk.Label(enterWindow, text='resize (%)').pack()
    enterA=tk.Entry(enterWindow)
    enterA.pack()
    btn = tk.Button(enterWindow, text='enter', command=doresize)     # 建立 Button 按鈕
    btn.pack() 

def doresize():
    global enterA,enterWindow,photo,frame
    a=enterA.get() #取得輸入框的資料
    a=float(a)/100
    if(a==1.0): #原圖的1倍不用調整
        enterWindow.destroy()
        return
    elif(a<=0): #不能有0倍或負的倍數
        enterWindow.destroy()
        return
    #print(a)
    w, h = photo.size
    newW = round(w*a) #圖新的大小
    newH = round(h*a)
    img = np.array(photo) #圖片轉陣列
    done = [[0] * (newW+1) for i in range(newH+1)] #存縮放好的陣列(x,y,value 的那個)
    img1 = [[],[],[]] #每行是一個點的資料, 第一列: x座標(把圖片中心移到(0,0)後的), 第二列:y座標(把圖片中心移到(0,0)後的), 第三列:那個點的值(0~255) 
    for i in range(h): #把圖片轉成img1並變成縮放後的座標
        for j in range(w):
            img1[0].append(j*a)
            img1[1].append(i*a)
            img1[2].append(img[i][j])
    img1 = np.array(img1).astype(int)

    for i in range(len(img1[0])): #換回 圖的陣列
        x=img1[0][i]
        y=img1[1][i]
        done[y][x]=img1[2][i]
    done = np.array(done)

    #雙線性差值
    for k in range(w*(h-1)-1): #跑過原圖所有點除了最後一列跟倒數第二列最後一行
        x0=img1[0][k]
        y0=img1[1][k]
        x1=img1[0][k+w+1]
        y1=img1[1][k+w+1]
        val00=img1[2][k]
        val10=img1[2][k+1]
        val01=img1[2][k+w]
        val11=img1[2][k+w+1]
        if(x0>x1):continue #x0是圖片中每列的最後一行時就跳過
        for i in range(y0,y1+1):
            for j in range(x0,x1+1):
                if(i in [y0,y1] and j in [x0,x1]):continue #(x0,y0)(x0,y1)(x1,y0)(x1,y1)這四點不用找值(已經有了)
                lunda = (i-y0)/(y1-y0) #代公式(跟上課ppt的x,y相反)
                mu = (j-x0)/(x1-x0)
                done[i][j] = lunda*mu*val11 + lunda*(1-mu)*val01 + (1-lunda)*mu*val10 + (1-lunda)*(1-mu)*val00 

    done = done.astype('uint8')
    photo = Image.fromarray(done)
    enterWindow.destroy()
    updata(newW, newH, photo, frame, 1)
    #print(photo)
    



## 旋轉(逆時針轉) ##
def rotation():
    global enterA,enterWindow
    enterWindow=tk.Toplevel(window) #產生子視窗來輸入數值
    enterWindow.title('parameter')
    enterWindow.geometry('380x150')
    tk.Label(enterWindow, text='degree (counterclockwise)').pack()
    enterA=tk.Entry(enterWindow)
    enterA.pack()
    btn = tk.Button(enterWindow, text='enter', command=dorotate)     # 建立 Button 按鈕
    btn.pack() 

def dorotate():
    global enterA,enterWindow,photo,frame
    a=enterA.get() #取得輸入框的資料 #角度
    a=int(a)%360
    #print(a)
    w, h = photo.size
    done = [[0] * w for i in range(h)] #存轉好的陣列(x,y,value 的那個)
    img = np.array(photo) #圖片轉陣列
    img1 = [[],[],[]] #每行是一個點的資料, 第一列: x座標(把圖片中心移到(0,0)後的), 第二列:y座標(把圖片中心移到(0,0)後的), 第三列:那個點的值(0~255) 
    for i in range(h): #把圖片轉成img1再進行矩陣運算
        for j in range(w):
            img1[0].append(j)
            img1[1].append(i)
            img1[2].append(img[i][j])
    img1 = np.array(img1)
    if(a==0):enterWindow.destroy() #不變
    else:
        middleX = w/2
        middleY = h/2
        img1[0] = img1[0]-middleX #把圖片中心的x,y座標移到(0,0)
        img1[1] = img1[1]-middleY
        rot = [[math.cos(math.radians(a)), math.sin(math.radians(a)), 0],[-1*math.sin(math.radians(a)), math.cos(math.radians(a)), 0],[0,0,1]] #旋轉矩陣
        img1 = np.dot(rot,img1).astype(int) #旋轉
        img1[0] = img1[0]+middleX #x,y座標移回來
        img1[1] = img1[1]+middleY
    

    for i in range(len(img1[0])): #換回 圖的陣列
        x=img1[0][i]
        y=img1[1][i]
        if(x<w and x>=0 and y<h and y>=0):
            done[y][x]=img1[2][i]
    
    done = np.array(done).astype('uint8')
    done = medianFilter(done,h,w) #條完角度的圖有的點會沒有值(像有雜訊一樣)，用median filter來修
    photo = Image.fromarray(done) #把矩陣轉回圖片
    enterWindow.destroy()
    updata(w, h, photo, frame, 1)
    

# median filter
def medianFilter(img,h,w): #輸入圖的矩陣,長,寬
    n=1 #取點周圍粗細幾元素的外圍來取中位數
    for i in range(1,h-1):
        for j in range(1,w-1):
            a=[]
            for ii in range(i-n,i+n+1):
                for jj in range(j-n,j+n+1):
                    a.append(img[ii][jj])
            a=np.array(a)
            img[i][j]=np.median(a)
    return img



## Gray-level slicing ##
def glSlicing():
    global enterA,enterB,val,enterWindow
    enterWindow=tk.Toplevel(window) #產生子視窗來輸入數值
    enterWindow.title('parameter')
    enterWindow.geometry('380x250')
    tk.Label(enterWindow, text='range(0~255):\n\nlower bound').pack()
    enterA=tk.Entry(enterWindow)
    enterA.pack()
    tk.Label(enterWindow, text='upper bound').pack()
    enterB=tk.Entry(enterWindow)
    enterB.pack()
    tk.Label(enterWindow, text=' \nDo you want to preserve the original values of unselected areas?').pack()
    val = tk.IntVar()
    rbtn1 = tk.Radiobutton(enterWindow, text='Yes',variable=val, value=1)
    rbtn1.pack()
    rbtn1.select()  # 自動選取yes選項
    rbtn2 = tk.Radiobutton(enterWindow, text='No',variable=val, value=0)
    rbtn2.pack()

    btn = tk.Button(enterWindow, text='enter', command=doglSlicing)     # 建立 Button 按鈕
    btn.pack() 

def doglSlicing():
    global enterA,enterB,val,enterWindow,photo,frame
    a=enterA.get() #取得輸入框的資料
    b=enterB.get()
    a=int(a)
    b=int(b)
    if(b<a or b>255 or a<0): #輸入的值不對結束執行
        enterWindow.destroy()
        return
    val1=val.get() #取得rbtn的資料
    img = np.array(photo) #圖片轉陣列
    w, h = photo.size
    if(val1): #沒選到的區域不變
        for i in range(h):
            for j in range(w):
                p=img[i][j]
                if(p>a and p<b):img[i][j]=255
    else:     #沒選到的區域變0
        for i in range(h):
            for j in range(w):
                p=img[i][j]
                if(p>a and p<b):img[i][j]=255
                else:img[i][j]=0

    img = img.astype('uint8')
    photo = Image.fromarray(img)
    enterWindow.destroy()
    updata(w, h, photo, frame, 1)
    #print(photo)
    


## bit plane ##
def bPlane():
    global val,enterWindow
    enterWindow=tk.Toplevel(window)
    enterWindow.title('parameter') #產生子視窗來輸入數值
    enterWindow.geometry('380x150')
    tk.Label(enterWindow, text='which bit plane').pack()

    optionList = ['1','2','3','4','5','6','7','8']
    val = tk.StringVar()
    val.set('1')
    menu = tk.OptionMenu(enterWindow, val, *optionList)  # 選單
    menu.config(fg='#f00')                # 設定樣式
    menu.pack()

    btn = tk.Button(enterWindow, text='enter', command=dobPlane)     # 建立 Button 按鈕
    btn.pack() 

def dobPlane():
    global val,enterWindow,photo,frame
    val1=int(val.get()) #取得要第幾位bit
    #print(val1)
    img = np.array(photo) #圖片轉陣列
    w, h = photo.size

    for i in range(h):
        for j in range(w):
            b='{:08b}'.format(img[i][j]) #把img[i][j]變成8位元的二進制字串
            if(b[8-val1]=='1'):img[i][j]=255
            else:img[i][j]=0

    img = img.astype('uint8')
    photo = Image.fromarray(img)
    enterWindow.destroy()
    updata(w, h, photo, frame, 1)
    #print(photo)
    


## smooth ##
def mysmooth():
    global spinbox,enterWindow
    enterWindow=tk.Toplevel(window) #產生子視窗來輸入數值
    enterWindow.title('parameter')
    enterWindow.geometry('380x150')
    tk.Label(enterWindow, text='degree of smoothness').pack()
    spinbox = tk.Spinbox(enterWindow, from_=1, to=10)
    spinbox.pack()

    #tk.Label(enterWindow, text=' ').pack()
    btn = tk.Button(enterWindow, text='enter', command=domysmooth)     # 建立 Button 按鈕
    btn.pack() 

def domysmooth():
    global spinbox,enterWindow,photo,frame
    val=int(spinbox.get()) #取得平滑程度(smoothm運算矩陣中心外圍幾元素)
    img = np.array(photo) #圖片轉陣列
    w, h = photo.size
    val1=(2*val+1)*(2*val+1) #取一次平均的點的個數
    img1 = [[0] * (w+2*val) for i in range(h+2*val)] #給圖的邊界多幾圈0來運算
    img1 = np.array(img1)
    for i in range(h): #把圖放進運算用矩陣
        for j in range(w):
            img1[i+val][j+val] = img[i][j]
    for i in range(h): #運算(每個點變成點周圍一圈寬度val個元素 的平均)
        for j in range(w):
            sum = 0
            for k in range(i,i+2*val+1):
                for m in range(j,j+2*val+1):
                    sum += img1[k][m]
            img[i][j] = sum/(val1)

    img = img.astype('uint8')
    photo = Image.fromarray(img)
    enterWindow.destroy()
    updata(w, h, photo, frame, 1)
    #print(photo)
    


## sharp ##
def mysharp():
    global spinbox,enterWindow
    enterWindow=tk.Toplevel(window) #產生子視窗來輸入數值
    enterWindow.title('parameter')
    enterWindow.geometry('380x150')
    tk.Label(enterWindow, text='degree of sharpness').pack()
    spinbox = tk.Spinbox(enterWindow, from_=1, to=10)
    spinbox.pack()

    btn = tk.Button(enterWindow, text='enter', command=domysharp)     # 建立 Button 按鈕
    btn.pack() 

def domysharp():
    global spinbox,enterWindow,photo,frame
    val=int(spinbox.get()) #取得銳利程度(sharp運算乘val倍)
    print(val)
    val=val*0.3
    img = np.array(photo).astype('float64') #圖片轉陣列
    w, h = photo.size
    done = img.copy() #存銳利化後的圖層(之後還要用原圖減掉)
    for i in range(1,h-1): #運算(每個點變成 (點的上下左右各個一點相加)*val再減掉原本點的4*val倍)
        for j in range(1,w-1):
            done[i][j] = (img[i-1][j]+img[i+1][j]+img[i][j-1]+img[i][j+1])*val - img[i][j]*4*val
    #done = np.clip(done,0,255).astype('uint8')
    img = img-done
    img = np.clip(img,0,255).astype('uint8') #clip(img,0,255) 超過255的當255小於0地當0 #.astype('uint8') 取整數
    #print(img, done)
    photo = Image.fromarray(img)
    enterWindow.destroy()
    updata(w, h, photo, frame, 1)
    #print(photo)
    


## auto-level (histogram equalization) ##
def al():
    global photo,frame
    w, h = photo.size
    img = np.array(photo)
    img1 = [[0] * w for i in range(h)] #存改完的圖
    img1 = np.array(img1)
    histo = np.array(count(img))
    #print(h)
    histo = histo/(w*h) #每個色階出現的機率
    #print(w,h)
    s=np.zeros(256,float)
    s[0]=histo[0]
    for i in range(1,256):
        s[i]=s[i-1]+histo[i] #色階跟數量的CDF
    s=s*255   #L-1=255                             (ex:   0,1,2,3,4,5
    s=np.around(s).astype('uint16') #四捨五入取整數     s=[0,0,1,2,3,5]  色階0變0,色階1變0,色階2變1,色階3變2,色階4變3,色階5還是5
    s[255]=255
    for i in range(h):
        for j in range(w):
            img1[i][j]=s[img[i][j]]
    
    img1 = img1.astype('uint8')
    photo = Image.fromarray(img1)
    updata(w, h, photo, frame, 1)

def count(img): #算每個色階有幾個點
    img1 = np.array(img) #圖片轉陣列
    c=np.zeros(256,int)
    for i in range(0,256):
        c[i]=np.sum(img1==i)
    return c





#設定使用者介面
window = tk.Tk()
window.title('B112040007-hw1')
window.geometry('800x1080') #1920x1080

menu1 = tk.Menu(window) #一直在最上面的選單

frame = tk.Frame(window, width=300, height=300) #用來放圖片
frame.pack()

filemenu = tk.Menu(menu1, tearoff=0) #tearoff=0 關閉子選單裡的虛線欄
menu1.add_cascade(label="File", menu=filemenu) #把file加進選單
filemenu.add_command(label="New file", command=open) #file的下拉選單
filemenu.add_command(label="Save", command=save)
filemenu.add_command(label="Save as...", command=saveAs)

menu1.add_command(label="Undo", command=undo) #回上一步 加進選單
menu1.add_command(label="Redo", command=redo) #取消復原 加進選單

tool1menu = tk.Menu(menu1, tearoff=0)
menu1.add_cascade(label="Contrast/Brightness", menu=tool1menu) #加進選單
tool1menu.add_command(label="Linear", command=linear) #下拉選單
tool1menu.add_command(label="Exponential", command=exponentail)
tool1menu.add_command(label="Logarithmical", command=logarithmical)

menu1.add_command(label="Change size", command=resize) 

menu1.add_command(label="Rotate", command=rotation)

menu1.add_command(label="Gray-level slicing", command=glSlicing) 

menu1.add_command(label="Auto-level", command=al) 

menu1.add_command(label="Bit-plane", command=bPlane) 

tool2menu = tk.Menu(menu1, tearoff=0)
menu1.add_cascade(label="Smooth & Sharpen", menu=tool2menu)
tool2menu.add_command(label="Smooth", command=mysmooth) 
tool2menu.add_command(label="Sharpen", command=mysharp) 

window.config(menu=menu1)  #顯示選單


 
window.mainloop() #使程式常駐執行，沒有這行會因為程式執行完什麼都不會看到，記得要打在程式的最後一行!


