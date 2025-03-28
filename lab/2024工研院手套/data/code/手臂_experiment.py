from sklearn.metrics import accuracy_score
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import joblib
import mne

import socket
import pygame
from pylsl import StreamInlet, resolve_stream
from pylsl import StreamInfo, StreamOutlet

from collections import defaultdict
from datetime import datetime
import subprocess
import time
import sys
import os
os.chdir(sys.path[0])#將當前環境位置設為當前"檔案位置"

def load_images():
  """載入實驗所需的圖像"""
  images = {
    # 載入 準備、開始、結束 的視圖
    "ready_pic"   : plt.imread('exp_img/ready_pic.jpg'),
    "fixation_pic": plt.imread('exp_img/fixation_pic.png'),
    "start_pic"   : plt.imread('exp_img/start_pic.jpg'),
    "end_pic"     : plt.imread('exp_img/end_pic.jpg'),
    # 載入正確和不正確 feedback 的視圖
    "true_pic"    : plt.imread('exp_img/true_pic.jpg'),
    "false_pic"   : plt.imread('exp_img/false_pic.jpg'),
    "late_pic"    : plt.imread('exp_img/late_pic.jpg'),
    "noisy_pic"   : plt.imread('exp_img/noisy_pic.png'),
    # 載入實驗倒計時 的視圖
    "init_count"  : [plt.imread(f'exp_img/init_count_{i:02d}_pic.jpg') for i in range(5,0,-1)],
    "task_count"  : [plt.imread(f'exp_img/task_count_{i:02d}_pic.jpg') for i in range(2,0,-1)],
    # 載入動作提示
    "rest_cue_pic" : plt.imread('exp_img/arm/arm_rest.png'),#手臂伸直
    "left_cue_pic" : plt.imread('exp_img/arm/left_arm_action.png'),#左手臂彎曲
    "right_cue_pic": plt.imread('exp_img/arm/right_arm_action.png') #右手臂彎曲
  }
  return images
  
def play_cue_sound(task):
  A_41 = "exp_img/arm/left.mp3"
  A_51 = "exp_img/arm/rest.mp3"
  A_61 = "exp_img/arm/right.mp3"
  if task == 41:
    pygame.mixer.init()
    pygame.mixer.music.load(A_41)
    pygame.mixer.music.play()
  elif task == 51:
    pygame.mixer.init()
    pygame.mixer.music.load(A_51)
    pygame.mixer.music.play()
  elif task == 61:
    pygame.mixer.init()
    pygame.mixer.music.load(A_61)
    pygame.mixer.music.play()
def phase_selection():
  """訓練/測試階段選擇"""
  while True:
    phase_select = input('請輸入實驗的階段:\n  1) 訓練階段\n  2) 測試階段\n')
    if phase_select=='1':
      return 'Train'
    elif phase_select=='2':
      return 'Test'
    print('輸入錯誤，請再次重試!!')
    print('\n'+'#'*50)
    
def arm_selection(arm):
  """選擇 左手臂/右手臂"""
  if arm == 'l':
    return (42, [41, 51])
  elif arm == 'r':
    return (62, [51, 61])
def trial_num_set():
  """訓練/測試 階段的試驗次數"""
  while True:
    trial_num = input('請輸入試驗次數：\n')
    try:
      return int(trial_num)
    except:
      pass
    print('輸入錯誤，請再次重試!!')
    print('\n'+'#'*50)
def model_list(path):
  files = [file for file in os.listdir(path) if '.png' not in file]
  groups = defaultdict(list)
  # 按檔名提取標籤並分類
  for file in files:
    sub_name, suffix = file.split('-kneebo_')
    groups[sub_name].append(suffix)
  # 檢查每組是否完整
  output = []
  output_dict = {}
  dict_count = 0
  for sub_name, suffixes in groups.items():
    left_files = [s for s in suffixes if '_l.' in s]
    right_files = [s for s in suffixes if '_r.' in s]
    # 如果所有模式都匹配，組合法有效
    if len(left_files)==5:
      output.append(f"左手臂【{sub_name}】")
      output_dict[dict_count]={'arm_name':'l','sub_name':sub_name}
      dict_count+=1
    if len(right_files)==5:
      output.append(f"右手臂【{sub_name}】")
      output_dict[dict_count]={'arm_name':'r','sub_name':sub_name}
      dict_count+=1
  # 給使用者選擇
  while True:
    print('輸入欲使用模型:')
    for idx, group in enumerate(output):
      arm_name = output_dict[idx]['arm_name']
      sub_name = output_dict[idx]['sub_name']
      fp = f'{path}/{sub_name}-kneebo_fs_psd_{arm_name}.npy'
      creation_time = os.path.getctime(fp)  # 獲取檔案建立時間（時間戳）
      formatted_time = datetime.fromtimestamp(creation_time).strftime('%Y/%m/%d %H:%M')
      print(f"  {idx}.{group}建立時間:{formatted_time}")
    try:
      model_sel = int(input())
      if model_sel not in output_dict:
        continue
    except:
      print('輸入錯誤，請再次重試!!')
      print('\n'+'#'*50)
      continue
    arm_name = output_dict[model_sel]['arm_name']
    sub_name = output_dict[model_sel]['sub_name']
    return sub_name,arm_name
def load_model(sub_name,arm_name,train_path='EEG Data/model'):
  """載入訓練數據和模型文件"""
  X_fs_psd_path = f'{train_path}/{sub_name}-kneebo_fs_psd_{arm_name}.npy'
  clf_psd_path  = f'{train_path}/{sub_name}-kneebo_trained_clf_psd_{arm_name}.joblib'
  clf_pcc_path  = f'{train_path}/{sub_name}-kneebo_trained_clf_pcc_{arm_name}.joblib'
  clf_csp_path  = f'{train_path}/{sub_name}-kneebo_trained_clf_csp_{arm_name}.joblib'
  csp_path      = f'{train_path}/{sub_name}-kneebo_trained_csp_{arm_name}.joblib'
  try:
    X_fs_psd = np.load(X_fs_psd_path)     # 載入 PSD 分類器要用已挑出的特徵
    clf_psd  = joblib.load(clf_psd_path)  # 載入 PSD (功率譜密度) 分類器
    clf_pcc  = joblib.load(clf_pcc_path)  # 載入 PCC (皮爾森相關係數) 分類器
    clf_csp  = joblib.load(clf_csp_path)  # 載入 CSP (共通空間模式) 分類器
    csp = joblib.load(csp_path)           # 對 EEG 數據進行 CSP 轉換用
    return X_fs_psd, clf_psd, clf_pcc, csp, clf_csp
  except FileNotFoundError:
    print('未找到模型文件')
    print('\n'+'#'*50)
    input()
    exit()
   
def init_lsl_inlets():
  # 運動偵測 w/ IMU & Mouse
  IMU_streams = resolve_stream('name', 'MouseButtons')
  IMU_inlet = StreamInlet(IMU_streams[0])
  IMU_sample_temp, start_lsl_timestp = IMU_inlet.pull_sample()
  print('\n'+'IMU & Mouse LSL Object initilized!!')
  return IMU_inlet

def init_lsl_EEG_inlets():
  # 運動偵測 w/ EEG
  EEG_streams = resolve_stream('type', 'EEG')
  EEG_inlet = StreamInlet(EEG_streams[0])
  EEG_sample_temp, start_lsl_timestp = EEG_inlet.pull_sample()
  print('\nEEG LSL Object initilized!!')
  return EEG_inlet

def init_EEG(EEG_inlet):
  # 初始化 EEG
  EEG_chunk, EEG_chunk_lsl_timestp = EEG_inlet.pull_chunk()
  if len(EEG_chunk[0]) == 8:
    EEG_ch_label = ['Fp1', 'Fp2', 'Fz', 'C3', 'C4', 'Pz', 'O1', 'O2']
    EEG_sprate = 1000
    nfft = 1024
    EEG_bandpass = [7, 30]
    vol_th = 200
    ch_th = 0.5
  elif len(EEG_chunk[0]) == 32:
    EEG_ch_label = ['Fp1', 'Fp2', 'AF3', 'AF4', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT7', 'FC3', 'FCz', 'FC4', 'FT8', 'T7',
                    'C3', 'Cz', 'C4', 'T8', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2']
    EEG_sprate = 500
    nfft = 512
    EEG_bandpass = [0.5, 30]
    vol_th = 50
    ch_th = 0.2
  return EEG_ch_label,EEG_sprate,nfft,EEG_bandpass,vol_th,ch_th

def init_MNE(EEG_ch_label, EEG_sprate):
  EEG_mne_info = mne.create_info(ch_names=EEG_ch_label, sfreq=EEG_sprate, ch_types='eeg', verbose=None)
  biosemi_montage = mne.channels.make_standard_montage('standard_1020')
  EEG_mne_info.set_montage(biosemi_montage)
  return EEG_mne_info

def init_UDP_socket():
  """初始化UDP Socket物件連接KneeBO Exoskeleton"""
  client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  print('\nKneeBO UDP Socket Object initilized!!')
  return client

def init_lsl_Event_Marker_inlets(event_stream_name):
  # Event Marker - LSL Outlet
  info = StreamInfo(event_stream_name, 'Markers', 1, 0, 'int32', event_stream_name.replace(' ', '_'))
  event_outlet = StreamOutlet(info)
  print('\nEvent Marker LSL Object initilized!!')
  event_outlet.push_sample([11])
  return event_outlet

def generate_task_order(cue_pic_dict, trial_num):
  # 生成並隨機排列動作提示序列
  cue_pic = np.array(list(cue_pic_dict.keys()))
  task_order_arr = np.tile(cue_pic, trial_num)
  # 打亂3次確保足夠亂
  np.random.shuffle(task_order_arr)
  np.random.shuffle(task_order_arr)
  np.random.shuffle(task_order_arr)
  return task_order_arr.tolist()

def get_EEG(EEG_inlet,EEG_buffer):
  # 將EEG資料放入 buffer
  EEG_chunk, EEG_chunk_lsl_timestp = EEG_inlet.pull_chunk()
  EEG_chunk = np.array(EEG_chunk).T
  try:
    EEG_buffer = np.hstack([EEG_buffer, EEG_chunk])
  except:
    EEG_buffer = EEG_chunk
  return EEG_buffer
  
  
class ExperimentFigures:
  def __init__(self, images):
    self.images = images
    self.task_count_pic_list = images['task_count']
    plt.ion()
    fig = plt.figure('BCI KneeBO System (by Lofan)', figsize=(8, 6))
    gs = gridspec.GridSpec(6, 2)
    plt.imshow(self.images['ready_pic'])
    plt.axis('off')
    fig.canvas.flush_events()
    
    # 設定為屬性，以便在方法中使用
    self.fig = fig
    self.gs = gs
    self.click = False

    # 註冊按鍵事件
    self.fig.canvas.mpl_connect('button_press_event', self.onclick)
  def disp_rest(self):
    plt.imshow(self.images['ready_pic'])
    plt.axis('off')
    self.fig.canvas.flush_events()
    self.click = False
    self.fig.canvas.mpl_connect('button_press_event', self.onclick)
    # 等待使用者按下按鈕
    self.wait_for_click()
    # 倒數計時
    self.countdown(3)
  # 等待使用者按下按鈕
  def wait_for_click(self):
    while not self.click:
      plt.pause(1)
  # 檢查是否按在按鈕範圍內
  def onclick(self, event):
    if event.xdata is not None and event.ydata is not None:
      if 590 <= event.xdata <= 1900 and 1700 <= event.ydata <= 2000:
        self.click = True
      else:
        self.click = False
  # 倒數計時
  def countdown(self,n=5):
    for init_count_pic in self.images['init_count'][5-n:]:
      plt.cla()
      plt.imshow(init_count_pic)
      plt.axis('off')
      self.fig.canvas.flush_events()
      plt.pause(1)
  # 顯示開始影像:
  def exp_start(self):
    plt.clf()
    plt.subplot(self.gs[1:, :])
    plt.imshow(self.images['start_pic'])
    plt.axis('off')
    self.fig.canvas.flush_events()
    plt.pause(3)
  # 顯示分數
  def disp_score(self,score,Title):
    plt.subplot(self.gs[0, 1])
    plt.cla()
    plt.title(Title)
    plt.text(0, 0.35, 'Score: %08d' % (score), fontsize=30, fontweight='light')
    plt.axis('off')
    self.fig.canvas.flush_events()
  # Fixation Cue
  def disp_fixation_cue(self):
    self.clear_main_fig()
    plt.imshow(self.images['fixation_pic'])
    plt.axis('off')
    self.fig.canvas.flush_events()
    print('Fixation:', 22)
    plt.pause(1)
  # Action Cue
  def disp_action_cue(self, cue_pic_dict, task_order):
    play_cue_sound(task_order)# <-----------------------------------
    self.clear_main_fig()
    plt.imshow(cue_pic_dict[task_order])
    plt.axis('off')
    self.fig.canvas.flush_events()
    print('Action Cue:', task_order)
  # Trial Countdown
  def disp_trial_countdown(self,task_cnt_i):
    self.clear_task_countdown()
    plt.imshow(self.task_count_pic_list[task_cnt_i])
    plt.axis('off')
    self.fig.canvas.flush_events()
  # noisy
  def disp_noisy(self):
    print('Trial Feedback:', 33, '(Noisy!!)')
    self.clear_task_countdown()
    self.clear_main_fig()
    plt.imshow(self.images['noisy_pic'])
    plt.axis('off')
    self.fig.canvas.flush_events()
  # 清除左上 task countdown
  def clear_task_countdown(self):
    plt.subplot(self.gs[0, 0])
    plt.cla()
    plt.axis('off')
  # 清除主視窗
  def clear_main_fig(self):
    plt.subplot(self.gs[1:, :])
    plt.cla()
    plt.axis('off')
    self.fig.canvas.flush_events()
  # 顯示本次試驗分析狀態
  def disp_task_status(self,status_pic,msg):
    self.clear_main_fig()
    plt.title(msg, fontsize=18, fontweight='bold')
    plt.imshow(status_pic)
    plt.axis('off')
    self.fig.canvas.flush_events()
  # end
  def disp_end(self):
    self.clear_main_fig()
    plt.imshow(self.images['end_pic'])
    plt.axis('off')
    self.fig.canvas.flush_events()
def BCI_KneeBO(event_stream_name='BCI - LSL Marker', kneebo_server_ip='127.0.0.6', kneebo_server_port=56950):
    ################################################################################
    # 載入實驗所需的圖像
    images = load_images()
    ################################################################################
    #訓練/測試 階段選擇
    phase_select = phase_selection()
    print('\n'+'#'*50)
    if phase_select == 'Test':
      model_path = 'EEG Data/model'
      sub_name,arm_name = model_list(path=model_path)
      arm_select, y_label = arm_selection(arm_name)
      print(arm_select, y_label)
      X_fs_psd, clf_psd, clf_pcc, csp, clf_csp = load_model(sub_name,arm_name,train_path=model_path)
      print('\n'+'#'*50)
    # Action Cue Figures - Loading
    cue_pic_dict = dict()
    if phase_select == 'Train':
      cue_pic_dict.update({51: images['rest_cue_pic']})
      cue_pic_dict.update({41: images['left_cue_pic']})
      cue_pic_dict.update({61: images['right_cue_pic']})
    elif phase_select == 'Test' and arm_name == 'r':
      cue_pic_dict.update({51: images['rest_cue_pic']})
      cue_pic_dict.update({61: images['right_cue_pic']})
    elif phase_select == 'Test' and arm_name == 'l':
      cue_pic_dict.update({51: images['rest_cue_pic']})
      cue_pic_dict.update({41: images['left_cue_pic']})
    # 訓練/測試 階段的試驗次數
    trial_num = trial_num_set()
    print('\n'+'#'*50)
    ################################################################################
    # 運動偵測 w/ IMU & Mouse
    subprocess.Popen('./mouse.exe')
    IMU_inlet = init_lsl_inlets()
    ################################################################################
    # Testing Phase
    if phase_select == 'Test':
      # 運動偵測 w/ EEG
      EEG_inlet = init_lsl_EEG_inlets()
      # 初始化 EEG & MNE Information
      plt.pause(0.1)
      EEG_ch_label,EEG_sprate,nfft,EEG_bandpass,vol_th,ch_th = init_EEG(EEG_inlet)
      EEG_mne_info = init_MNE(EEG_ch_label, EEG_sprate)
      # 初始化 UDP Socket物件連接 KneeBO Exoskeleton
      # kneebo_client_obj = init_UDP_socket()

    # Event Marker - LSL Outlet
    event_outlet = init_lsl_Event_Marker_inlets(event_stream_name)
    
    # Action Cue List - Arrangement
    # 生成並隨機排列動作提示序列
    task_order_list = generate_task_order(cue_pic_dict, trial_num)

    ################################################################################
    # Experiment Scene - Execution
    exp_fig = ExperimentFigures(images)
    # 等待使用者按下按鈕
    exp_fig.wait_for_click()
    # 倒數計時
    exp_fig.countdown()
    
    # 實驗開始
    event_outlet.push_sample([11]) # Event Marker lsl 
    exp_fig.exp_start()

    # Score - Initilization
    score = 0
    exp_fig.disp_score(score,'')

    # Tasks - Start
    y_test = []
    y_predict = []
    # task_i = 0
    for task_i,task_order in enumerate(task_order_list):
      if task_i!=0 and task_i%30==0:#每10組休息一次
        exp_fig.disp_rest()
      print('Trial No.', task_i+1)
      trial_statistics = f'{task_i+1}/{len(task_order_list)}'
      # Fixation Cue
      event_outlet.push_sample([22])
      exp_fig.disp_fixation_cue()

      # LSL Streaming - Flush
      IMU_chunk, IMU_lsl_timestp = IMU_inlet.pull_chunk()#應該可以移除?
      #持續拉取 EEG 資料，並檢查每次拉取的資料塊大小。
      #如果資料塊的行數小於 25，則認為資料不足並退出迴圈。
      while phase_select == 'Test':
        EEG_chunk, EEG_lsl_timestp = EEG_inlet.pull_chunk()
        EEG_chunk = np.array(EEG_chunk)
        if EEG_chunk.shape[0] < 25:
          break
      EEG_buffer = []
      X_epoch = []

      # Action Cue
      event_outlet.push_sample([task_order])
      exp_fig.disp_action_cue(cue_pic_dict, task_order)

      # Trial Countdown - Initialization
      feedback_sample = None
      task_start = time.time()
      task_cnt_i = 0
      task_time = len(images['task_count']) # 2秒
      task_switch_list = [True] * (task_time+1)
      while time.time() - task_start < task_time:# ?
        # Trial Countdown
        if time.time() - task_start > task_cnt_i and task_switch_list[task_cnt_i] == True:
          exp_fig.disp_trial_countdown(task_cnt_i)
          task_switch_list[task_cnt_i] = False
          if task_cnt_i < task_time-1:
            task_cnt_i += 1

        # Testing Phase - Motor Detection w/ EEG
        if phase_select == 'Test':
          plt.pause(0.2)
          EEG_buffer = get_EEG(EEG_inlet,EEG_buffer)
          # 取出EEG 訊號的前 1.2秒 預處理
          if EEG_buffer.shape[1] > EEG_sprate * 1.2:
            X_epoch = EEG_epoch_preprocess(EEG_buffer, EEG_mne_info, EEG_ch_label, sprate=EEG_sprate, bandpass=EEG_bandpass, vol_th=vol_th, ch_th=ch_th)
            if len(X_epoch) == 0:
              feedback_sample = 'noisy'
            else:
              # 判斷當前的 EEG 訊號對應的輸出
              feedback_sample = EEG_ML_KneeBO(
                  X_epoch[np.newaxis], y_label, X_fs_psd, clf_psd, clf_pcc, csp, clf_csp, sprate=EEG_sprate, nfft=nfft, bandpass=EEG_bandpass)
              y_test.append(task_order)
              feedback_to_predict = {42: 41, None: 51, 62: 61}
              y_predict.append(feedback_to_predict[feedback_sample])

        # Training Phase - Motor Detection w/ IMU & Mouse
        # 抓取滑鼠資料，檢測滑鼠是否動作，若按下左鍵，則輸出左腳、右鍵輸出右腳，滾輪輸出resting，若什麼都不按輸出正確解
        IMU_chunk, IMU_lsl_timestp = IMU_inlet.pull_chunk()
        if len(IMU_chunk) > 0:
          feedback_sample = IMU_chunk[-1][0]
          if feedback_sample[11:15] == 'Left':
            feedback_sample = 42
          elif feedback_sample[11:16] == 'Right':
            feedback_sample = 62
          elif feedback_sample[5:10] == 'Wheel':
            feedback_sample = None
        elif phase_select == 'Train' and time.time() - task_start > 1.5:
          if task_order == 41:
            feedback_sample = 42
          elif task_order == 61:
            feedback_sample = 62

        # Motor Detection - Figure Feedback
        # No Any Detection
        if (phase_select == 'Train' and feedback_sample == None) or (phase_select == 'Test' and len(X_epoch) == 0 and feedback_sample == None):
          continue

        elif phase_select == 'Test' and feedback_sample == 'noisy':
          event_outlet.push_sample([33])
          task_order_list.append(task_order)
          exp_fig.disp_noisy()
          break
        elif phase_select == 'Test' and len(X_epoch) > 0 and feedback_sample == None:
          break

        # Motor Task - Correct
        elif feedback_sample == task_order+1:
          event_outlet.push_sample([task_order+1])
          print('Trial Feedback:', task_order+1, '(Correct!!)')

          if phase_select == 'Train':
            score_temp = 100 * (1 - round((time.time() - task_start)/task_time, 1))
          else:
            score_temp = 100
          score += score_temp
          exp_fig.clear_task_countdown()
          exp_fig.disp_score(score,trial_statistics)
          exp_fig.disp_task_status(images['true_pic'], f'+ {score_temp}')

          # KneeBO Feedback
          if phase_select == 'Test' and feedback_sample == arm_select:
            print('move2')
            # kneebo_client_obj.sendto('move2'.encode(), (kneebo_server_ip, kneebo_server_port))
            plt.pause(5)
          break

        # Motor/Rest Task - Incorrect (Wrong Action)
        else:
          event_outlet.push_sample([33])
          print('Trial Feedback:', 33, '(Wrong!!)')
          task_order_list.append(task_order)
          
          score -= 100
          exp_fig.clear_task_countdown()
          exp_fig.disp_score(score,trial_statistics)
          exp_fig.disp_task_status(images['false_pic'], '- 100')
          break
      ################################################################################
      # Rest Task - Correct
      if task_order == 51 and feedback_sample == None:
        event_outlet.push_sample([53])
        print('Trial Feedback:', 53, '(Correct!!)')
        score += 100
        exp_fig.clear_task_countdown()
        exp_fig.disp_score(score,trial_statistics)
        exp_fig.disp_task_status(images['true_pic'], '+ 100')

      # Motor Task - Incorrect (Late)
      elif feedback_sample == None:
        event_outlet.push_sample([33])
        print('Trial Feedback:', 33, '(Late!!)')
        task_order_list.append(task_order)
        
        score -= 100
        exp_fig.clear_task_countdown()
        exp_fig.disp_score(score,trial_statistics)
        exp_fig.disp_task_status(images['late_pic'], '- 100')
        
      print('\n'+'#'*50)
      plt.pause(1)
      exp_fig.clear_main_fig()
      plt.pause(1)
    
    exp_fig.disp_end()
    event_outlet.push_sample([99])
    if phase_select == 'Test':
      #kneebo_client_obj.close()
      print('Accuracy:', accuracy_score(y_test, y_predict))
      np.save('EEG Data'+'/'+sub_name+'-kneebo_y_test_'+arm_name+'.npy', y_test)
      np.save('EEG Data'+'/'+sub_name+'-kneebo_y_predict_'+arm_name+'.npy', y_predict)

    print('='*50+'\n\nThe experiment is over, the window will close in 3 seconds!!')
    plt.pause(3)
    plt.ioff()


def IIR_filter(X_epoch=[], fre_cutoff=[0.5, 50], sprate=500, axis=0, pass_type='bandpass', ftype='butter', filter_order=10, filter_plot=False, ch_label=[]):
  # IIR filter - filter design
  sos = signal.iirfilter(filter_order, Wn=fre_cutoff, btype=pass_type, analog=False, ftype=ftype, output='sos', fs=sprate)

  # IIR filter - signal bandpass
  if len(X_epoch) > 0:
    X_epoch_filtered = signal.sosfiltfilt(sos, X_epoch, axis=axis)

  # IIR filter - filter performance plot (optional)
  if filter_plot == True:
    f_axis, h = signal.sosfreqz(sos, 512, fs=sprate)
    plt.figure(figsize=(12, 9))

    if len(X_epoch) > 0 and X_epoch.ndim <= 3:
      plt.subplot(311)
    plt.plot(f_axis, 20 * np.log10(np.maximum(abs(h), 1e-5)))
    try:
      plt.plot([fre_cutoff[0], fre_cutoff[0]], [-105, 5], '--', label=str(fre_cutoff[0])+' Hz')
      plt.plot([fre_cutoff[1], fre_cutoff[1]], [-105, 5], '--', label=str(fre_cutoff[1])+' Hz')
      plt.xticks(np.hstack([np.arange(0, fre_cutoff[1]+20, 5), fre_cutoff[0], fre_cutoff[1]]))
      plt.xlim(0, fre_cutoff[1]+20)
    except:
      plt.plot([fre_cutoff, fre_cutoff], [-105, 5], '--', label=str(fre_cutoff)+' Hz')
      plt.xticks(np.hstack([np.arange(0, fre_cutoff+20, 5), fre_cutoff]))
      plt.xlim(0, fre_cutoff+20)
    plt.yticks(np.arange(-100, 10, 10))
    plt.ylim(-100, 10)
    plt.title('IIR ('+ftype+') Filter frequency response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Quantity (dB)')
    plt.legend(loc='lower right')
    plt.grid()

    if len(X_epoch) > 0 and X_epoch.ndim <= 3:
      if len(ch_label) == 0 and X_epoch.ndim >= 2:
        ch_label = ['Ch'+str(ch) for ch in range(X_epoch.shape[1])]

      t_axis = np.arange(0, X_epoch.shape[axis]/sprate, 1/sprate)
      if X_epoch.ndim == 3:
        X_epoch = X_epoch.mean(axis=0).T
        X_epoch_filtered_plot = X_epoch_filtered.mean(axis=0).T
      else:
        X_epoch_filtered_plot = X_epoch_filtered

      plt.subplot(312)
      plt.plot(t_axis, X_epoch, label=ch_label)
      plt.title('before filtering')
      plt.xlabel('Time (sec)')
      plt.ylabel('Voltage (μV)')
      plt.xlim(t_axis.min(), t_axis.max())
      if len(ch_label) <= 10:
        plt.legend(loc='lower right')
      plt.grid()

      plt.subplot(313)
      plt.plot(t_axis, X_epoch_filtered_plot, label=ch_label)
      plt.title('after filtering')
      plt.xlabel('Time (sec)')
      plt.ylabel('Voltage (μV)')
      plt.xlim(t_axis.min(), t_axis.max())
      if len(ch_label) <= 10:
        plt.legend(loc='lower right')
      plt.grid()
    plt.tight_layout()
    plt.show()

  if len(X_epoch) > 0:
    return X_epoch_filtered


def EEG_epoch_preprocess(EEG_epoch, EEG_mne_info, EEG_ch_label, sprate=500, bandpass=[0.5, 30], vol_th=50, ch_th=0.2):
  # Bandpass Filter
  EEG_epoch_ini = IIR_filter(EEG_epoch[:, :int(sprate*1.2)], fre_cutoff=bandpass, sprate=sprate, axis=1)
  EEG_epoch_filtered = EEG_epoch_ini[:, int(sprate*0.5):int(sprate*1)]
  print('Bandpass', EEG_epoch_filtered.shape)

  # Detect Bad Trial
  bad_epoch_ch, bad_epoch_sp = np.where((EEG_epoch_filtered < -vol_th) | (vol_th < EEG_epoch_filtered))
  if len(np.unique(bad_epoch_ch)) > EEG_epoch.shape[0] * ch_th:
    return []

  # Interpolate Bad Channels - 1
  EEG_mne_info['bads'] = []
  for ch_i in range(EEG_epoch_filtered.shape[0]):
    ch_max = np.abs(EEG_epoch_filtered[ch_i, :]).max()
    if ch_max > vol_th and EEG_ch_label[ch_i] not in EEG_mne_info['bads']:
      print(EEG_ch_label[ch_i], '- amp:', ch_max)
      EEG_mne_info['bads'].append(EEG_ch_label[ch_i])
  if len(EEG_mne_info['bads']) > 0:
    EEG_epoch_mne = mne.EvokedArray(EEG_epoch_filtered, EEG_mne_info)
    print(EEG_mne_info['bads'])
    EEG_epoch_mne.interpolate_bads()
    EEG_epoch_filtered = EEG_epoch_mne.data
  EEG_bad_ch = EEG_mne_info['bads']


  return EEG_epoch_filtered


def ML_X_corr(X_epoch):
  X_corr = []

  for X_epoch_single in X_epoch:
    EEG_corr = np.corrcoef(X_epoch_single)
    EEG_corr_triu_i = np.triu_indices_from(EEG_corr, k=1)

    X_corr.append(EEG_corr[EEG_corr_triu_i])
  X_corr = np.vstack(X_corr)

  return X_corr

# 在測試階段用，用於判斷當前的 EEG 訊號對應的輸出
def EEG_ML_KneeBO(X_epoch, y_label, X_fs_psd, clf_psd, clf_pcc, csp, clf_csp, sprate=500, nfft=512, bandpass=[0.5, 30]):
  # 1. 功率譜密度 PSD(Power Spectral Density)
  f_axis_welch, X_welch = signal.welch(X_epoch, sprate, window='hamming', nperseg=sprate//2, noverlap=sprate//4, nfft=nfft, axis=2)
  f_axis_welch_i, = np.where((bandpass[0] <= f_axis_welch) & (f_axis_welch <= bandpass[1]))
  X_psd = X_welch[:, :, f_axis_welch_i].reshape(X_welch.shape[0], X_welch.shape[1]*f_axis_welch_i.shape[0])
  y_proba_psd = clf_psd.predict_proba(X_psd[:, X_fs_psd])

  # 2. 皮爾森相關係數 PCC (Pearson Correlation Coefficient)
  X_pcc = ML_X_corr(X_epoch)
  y_proba_pcc = clf_pcc.predict_proba(X_pcc)

  # 3. 共通空間模式 CSP (Common Spatial Pattern)
  X_csp = csp.transform(X_epoch)
  y_proba_csp = clf_csp.predict_proba(X_csp)

  # Mix
  y_proba = (y_proba_psd + y_proba_pcc + y_proba_csp).squeeze()
  y_predict = y_label[y_proba.argmax()]

  if y_predict == 41:
    return 42
  elif y_predict == 61:
    return 62
  elif y_predict == 51:
    return None

class Test_function(object):
  def __init__(self):
    pass
  # 測試實驗用視窗
  def test_ExperimentFigures(self):
    images = load_images()
    exp_fig = ExperimentFigures(images)
    exp_fig.wait_for_click()
    # exp_fig.countdown()
    exp_fig.exp_start()
    exp_fig.disp_score(0,'')
    
    cue_pic_dict = dict()
    cue_pic_dict.update({51: images['rest_cue_pic']})
    cue_pic_dict.update({41: images['left_cue_pic']})
    cue_pic_dict.update({61: images['right_cue_pic']})
    trial_num = 9
    task_order_list = generate_task_order(cue_pic_dict, trial_num)
    
    print(task_order_list)
    input()
    score = 0
    for i,task_order in enumerate(task_order_list):
      trial_statistics = f'{i+1}/{len(task_order_list)}'
      if i!=0 and i%3==0:
        exp_fig.disp_rest()
      
      exp_fig.disp_fixation_cue()
      plt.pause(1)
      
      exp_fig.disp_action_cue(cue_pic_dict, task_order)
      for task_cnt_i in range(2):
        exp_fig.disp_trial_countdown(task_cnt_i)
        plt.pause(1)
      
      exp_fig.clear_task_countdown()
      
      if i%3==0:
        exp_fig.disp_task_status(images['true_pic'], f'+ {1000}')
        score += 1000
      if i%3==1:
        exp_fig.disp_task_status(images['late_pic'], '- 100')
        score -= 100
      if i%3==2:
        exp_fig.disp_task_status(images['false_pic'], '- 100')
        score -= 100
      exp_fig.disp_score(score,trial_statistics)
      #exp_fig.disp_noisy()
      print(score)
      plt.pause(1)
      exp_fig.clear_main_fig()
      plt.pause(1)
    exp_fig.disp_end()
  # 測試 IMU 滑鼠控制
  def test_mouse(self):
    subprocess.Popen('./mouse.exe')
    print('='*20)
    IMU_inlet = init_lsl_inlets()
    print(IMU_inlet)
    print('='*20)
    while True:
      IMU_chunk, IMU_lsl_timestp = IMU_inlet.pull_chunk()
      print(IMU_chunk)
      time.sleep(1)
  # 測試外骨骼連線狀態(需手動開啟，KneeBO_control_server.py 、 設備請先設定完成)
  def test_KneeBO_control(self):
    server_ip='127.0.0.6'
    port=56950
    udp_client_obj = init_UDP_socket()
    while True:
      udp_client_CMD = input("請輸入對 KneeBO 外骨骼的指令\n1) 0 - 120（°，KneeBo 角度）\n2) move \n3) move2\n4) end\n================================\n")
      print()
      # Send the command to the KneeBO Exoskeleton
      if udp_client_CMD == 'move' or udp_client_CMD == 'move2' or udp_client_CMD == 'flush' or udp_client_CMD == 'end':
        udp_client_obj.sendto(udp_client_CMD.encode(), (server_ip, port))
        if udp_client_CMD == 'end':
          break
      else:
        try:
          number = int(udp_client_CMD)
          udp_client_obj.sendto(udp_client_CMD.encode(), (server_ip, port))
        except:
          udp_client_obj.sendto('flush'.encode(), (server_ip, port))
          print("您輸入的指令有誤，請重試！")

    # Deconstruct the UDP Socket object
    udp_client_obj.close()
    
if __name__ == '__main__':

  BCI_KneeBO()
  # test = Test_function()
  # test.test_ExperimentFigures()
  # test.test_mouse()
  # test.test_KneeBO_control()
  

      

