import numpy as np
from scipy import signal
from sklearn.metrics import accuracy_score
import mne
import joblib

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import socket
from pylsl import StreamInlet, resolve_stream
from pylsl import StreamInfo, StreamOutlet

import os
import subprocess
import time

import sys
os.chdir(sys.path[0])#將當前環境位置設為當前"檔案位置"

import pygame

''' 
event_mark:
51:rest_cue
53:rest_end

411:left_hand_action_1
412:left_hand_action_1_end
421:left_hand_action_2
422:left_hand_action_2_end

611:right_hand_action_1
612:right_hand_action_1_end
621:right_hand_action_2
622:right_hand_action_2_end
'''
cue_to_feedback={
  411:412,
  421:422,
  611:612,
  621:622,
  51:None
}
feedback_to_cue={
  412:411,
  422:421,
  612:611,
  622:621,
  None:51
}
# 將event mark轉換成人便於理解的語言
feedback_to_msg = {
  411:"左手握拳",
  421:"左手張開手掌",
  611:"右手握拳",
  621:"右手張開手掌",
  51 :"休息"
}
def show_state(feedback):
  print(f'  當前動作:{feedback_to_msg[feedback]}')
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
    "rest_cue_pic" : plt.imread('exp_img/rest_cue_pic.jpg'),
    "left_cue_pic1" : plt.imread('exp_img/hand/left_hand_action_1.png'), #左手握拳
    "left_cue_pic2" : plt.imread('exp_img/hand/left_hand_action_2.png'), #左手張開手掌
    "right_cue_pic1": plt.imread('exp_img/hand/right_hand_action_1.png'),#右手握拳
    "right_cue_pic2": plt.imread('exp_img/hand/right_hand_action_2.png') #右手張開手掌
  }
  return images
def play_cue_sound(task):
  A_411 = "exp_img/hand/left_action1.mp3"
  A_421 = "exp_img/hand/left_action2.mp3"
  A_51 = "exp_img/hand/rest.mp3"
  A_611 = "exp_img/hand/right_action1.mp3"
  A_621 = "exp_img/hand/right_action2.mp3"
  if task == 411:
    pygame.mixer.init()
    pygame.mixer.music.load(A_411)
    pygame.mixer.music.play()
  if task == 421:
    pygame.mixer.init()
    pygame.mixer.music.load(A_421)
    pygame.mixer.music.play()
  elif task == 51:
    pygame.mixer.init()
    pygame.mixer.music.load(A_51)
    pygame.mixer.music.play()
  elif task == 611:
    pygame.mixer.init()
    pygame.mixer.music.load(A_611)
    pygame.mixer.music.play()
  elif task == 621:
    pygame.mixer.init()
    pygame.mixer.music.load(A_621)
    pygame.mixer.music.play()
def phase_selection():
  """訓練/測試階段選擇"""
  while True:
    phase_select = input('Please input the phase of system:\n1) Training Phase\n2) Testing Phase\n')
    if phase_select=='1':
      return 'Train'
    elif phase_select=='2':
      return 'Test'
    print('Input error, please try again!!')
    print('\n'+'#'*50)
    
def leg_selection():
  """選擇 左手/右手"""
  while True:
    leg_select = input('Please select the side of leg:\n1) Left Leg\n2) Right Leg\n')
    if leg_select == '1':
      return ('l', [411,421,51])
    elif leg_select == '2':
      return ('r', [611,621,51])
    print('Input error, please try again!!')
    print('\n'+'#'*50)
    
def trial_num_set():
  """訓練/測試 階段的試驗次數"""
  while True:
    trial_num = input('Please input trial number of each event:\n')
    try:
      return int(trial_num)
    except:
      pass
    print('Input error, please try again!!')
    print('\n'+'#'*50)
    
def load_model(leg_name,train_path='EEG Data'):
  """載入訓練數據和模型文件"""
  sub_name = input('Please input the filename of Training data:\n')
  sub_name = 'Artise 32ch - EEG sample file' if sub_name=='' else sub_name
  
  X_fs_psd_path = f'{train_path}/{sub_name}-kneebo_fs_psd_{leg_name}.npy'
  clf_psd_path  = f'{train_path}/{sub_name}-kneebo_trained_clf_psd_{leg_name}.joblib'
  clf_pcc_path  = f'{train_path}/{sub_name}-kneebo_trained_clf_pcc_{leg_name}.joblib'
  clf_csp_path  = f'{train_path}/{sub_name}-kneebo_trained_clf_csp_{leg_name}.joblib'
  csp_path      = f'{train_path}/{sub_name}-kneebo_trained_csp_{leg_name}.joblib'
  try:
    X_fs_psd = np.load(X_fs_psd_path)     # 載入 PSD 分類器要用已挑出的特徵
    clf_psd  = joblib.load(clf_psd_path)  # 載入 PSD (功率譜密度) 分類器
    clf_pcc  = joblib.load(clf_pcc_path)  # 載入 PCC (皮爾森相關係數) 分類器
    clf_csp  = joblib.load(clf_csp_path)  # 載入 CSP (共通空間模式) 分類器
    csp = joblib.load(csp_path)           # 對 EEG 數據進行 CSP 轉換用
    return X_fs_psd, clf_psd, clf_pcc, csp, clf_csp
  except FileNotFoundError:
    print('Model files not found, please try again.')
    print('\n'+'#'*50)
    return load_model(leg_name, train_path)
   
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

# def init_UDP_socket():
  # """初始化UDP Socket物件連接KneeBO Exoskeleton"""
  # client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  # print('\nKneeBO UDP Socket Object initilized!!')
  # return client

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
  # 打亂2次確保足夠亂
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
  def countdown(self):
    for init_count_pic in self.images['init_count']:
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
  def disp_score(self,score):
    plt.subplot(self.gs[0, 1])
    plt.cla()
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
    play_cue_sound(task_order)# <---------------------------
    self.clear_main_fig()
    plt.imshow(cue_pic_dict[task_order])
    plt.axis('off')
    self.fig.canvas.flush_events()
    # print('Action Cue:', task_order)
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
def BCI_KneeBO(event_stream_name='BCI - LSL Marker'):
    ################################################################################
    # 載入實驗所需的圖像
    images = load_images()
    ################################################################################
    #訓練/測試 階段選擇
    phase_select = phase_selection()
    print('\n'+'#'*50)
    if phase_select == 'Test':
      leg_name, y_label = leg_selection()
      print('\n'+'#'*50)
      X_fs_psd, clf_psd, clf_pcc, csp, clf_csp = load_model(leg_name,train_path='EEG Data')
      print('\n'+'#'*50)
    # Action Cue Figures - Loading
    cue_pic_dict = dict()
    if phase_select == 'Train':
      cue_pic_dict.update({51: images['rest_cue_pic']})
      cue_pic_dict.update({411: images['left_cue_pic1']})
      cue_pic_dict.update({421: images['left_cue_pic2']})
      cue_pic_dict.update({611: images['right_cue_pic1']})
      cue_pic_dict.update({621: images['right_cue_pic2']})
    elif phase_select == 'Test' and leg_name == 'r':
      cue_pic_dict.update({51: images['rest_cue_pic']})
      cue_pic_dict.update({611: images['right_cue_pic1']})
      cue_pic_dict.update({621: images['right_cue_pic2']})
    elif phase_select == 'Test' and leg_name == 'l':
      cue_pic_dict.update({51: images['rest_cue_pic']})
      cue_pic_dict.update({411: images['left_cue_pic1']})
      cue_pic_dict.update({421: images['left_cue_pic2']})
    # 訓練/測試 階段的試驗次數
    trial_num = trial_num_set()
    print('\n'+'#'*50)
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
    exp_fig.disp_score(score)

    # Tasks - Start
    y_test = []
    y_predict = []
    # task_i = 0
    for task_i,task_order in enumerate(task_order_list):
      print('Trial No.', task_i+1)
      
      # Fixation Cue
      event_outlet.push_sample([22])
      exp_fig.disp_fixation_cue()

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
              y_predict.append(feedback_to_cue[feedback_sample]) # 412->411、612->611

        # Training Phase - Motor Detection w/
        if phase_select == 'Train' and time.time() - task_start > 1.5:
          if task_order!=51 and (task_order in cue_to_feedback):
            feedback_sample = cue_to_feedback[task_order]# 411->412、611->612

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
        elif feedback_sample == cue_to_feedback[task_order]:#411->412
          event_outlet.push_sample([cue_to_feedback[task_order]])
          print('Trial Feedback:', cue_to_feedback[task_order], '(Correct!!)')

          if phase_select == 'Train':
            score_temp = 100 * (1 - round((time.time() - task_start)/task_time, 1))
          else:
            score_temp = 100
          score += score_temp
          exp_fig.clear_task_countdown()
          exp_fig.disp_score(score)
          exp_fig.disp_task_status(images['true_pic'], f'+ {score_temp}')

          if phase_select == 'Test':
            show_state(feedback_sample)
            plt.pause(5)
          break

        # Motor/Rest Task - Incorrect (Wrong Action)
        else:
          event_outlet.push_sample([33])
          print('Trial Feedback:', 33, '(Wrong!!)')
          task_order_list.append(task_order)
          
          score -= 100
          exp_fig.clear_task_countdown()
          exp_fig.disp_score(score)
          exp_fig.disp_task_status(images['false_pic'], '- 100')
          break
      ################################################################################
      # Rest Task - Correct
      if task_order == 51 and feedback_sample == None:
        event_outlet.push_sample([53])
        print('Trial Feedback:', 53, '(Correct!!)')
        score += 100
        exp_fig.clear_task_countdown()
        exp_fig.disp_score(score)
        exp_fig.disp_task_status(images['true_pic'], '+ 100')

      # Motor Task - Incorrect (Late)
      elif feedback_sample == None:
        event_outlet.push_sample([33])
        print('Trial Feedback:', 33, '(Late!!)')
        task_order_list.append(task_order)
        
        score -= 100
        exp_fig.clear_task_countdown()
        exp_fig.disp_score(score)
        exp_fig.disp_task_status(images['late_pic'], '- 100')
        
      print('\n'+'#'*50)
      plt.pause(1)
      exp_fig.clear_main_fig()
      plt.pause(1)
    
    exp_fig.disp_end()
    event_outlet.push_sample([99])
    if phase_select == 'Test':
      kneebo_client_obj.close()
      print('Accuracy:', accuracy_score(y_test, y_predict))
      np.save(train_path+'/'+sub_name+'-kneebo_y_test_'+leg_name+'.npy', y_test)
      np.save(train_path+'/'+sub_name+'-kneebo_y_predict_'+leg_name+'.npy', y_predict)

    print('='*50+'\n\nThe experiment is over, the window will close in 30 seconds!!')
    plt.pause(30)
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

  return cue_to_feedback[y_predict]
  # if y_predict == 411:
    # return 412
  # elif y_predict == 421:
    # return 422
  # elif y_predict == 611:
    # return 612
  # elif y_predict == 621:
    # return 622
  # elif y_predict == 51:
    # return None

class Test_function(object):
  def __init__(self):
    pass
  # 測試實驗用視窗
  def test_ExperimentFigures(self):
    print("當前為測試實驗影像")
    images = load_images()
    exp_fig = ExperimentFigures(images)
    exp_fig.wait_for_click()
    exp_fig.countdown()
    exp_fig.exp_start()
    exp_fig.disp_score(0)
    
    cue_pic_dict = dict()
    cue_pic_dict.update({51: images['rest_cue_pic']})
    cue_pic_dict.update({411: images['left_cue_pic1']})
    cue_pic_dict.update({421: images['left_cue_pic2']})
    cue_pic_dict.update({611: images['right_cue_pic1']})
    cue_pic_dict.update({621: images['right_cue_pic2']})
    task_order_list = generate_task_order(cue_pic_dict, 3)
    score = 0
    for i,task_order in enumerate(task_order_list):
      exp_fig.disp_fixation_cue()
      plt.pause(1)
      show_state(task_order)
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
      exp_fig.disp_score(score)
      print(score)
      plt.pause(1)
      exp_fig.clear_main_fig()
      plt.pause(1)
    #end
    exp_fig.disp_end()
    plt.pause(3)
    plt.ioff()
    
if __name__ == '__main__':
  # BCI_KneeBO()
  test = Test_function()
  test.test_ExperimentFigures()
  # test.test_KneeBO_control()
  

      

