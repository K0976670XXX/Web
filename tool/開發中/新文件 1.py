import math
from icecream import ic
left_mine_blue = 10000;
start_days = 10;
start_mag = 2;

start_mine_blue = 1000;

def proc_start_mag(left,days,mag,start_mine):
  total_start_mine = days*mag*start_mine
  after_left_mine = left-total_start_mine
  if (after_left_mine>0):
    return (after_left_mine/start_mine)+days
  else:
    return left/(mag*start_mine)

D = proc_start_mag(left_mine_blue,start_days,start_mag,525)
print(D)
# D = proc_start_mag(left_mine_blue,start_days,start_mag,500)
# print(D)
