import pandas as pd
from pylance import pandas 
data = pd.read_excel('Book1.xlsx')

salary = int(input("請輸入薪資總額："))
bonus = int(input("請輸入工作獎金："))
t = int(input("請輸入加班時數："))
d = int(input("請輸入病假天數："))
a = int(input("請輸入事假天數："))


totalsalary = salary + bonus + salary/240*t - d*salary/240*0.5 - a*salary/240

print("本月實領金額："+str(totalsalary))
