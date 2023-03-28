import pandas as pd

# 定義函數，計算一位員工的薪水
def calculate_pay(hourly_rate, overtime_hours, vacation_hours):
    # 計算基本工資
    if overtime_hours > 40:
        regular_pay = 40 * hourly_rate
        overtime_pay = (overtime_hours - 40) * hourly_rate * 1.5
    else:
        regular_pay = overtime_hours * hourly_rate
        overtime_pay = 0

    # 計算薪水
    gross_pay = regular_pay + overtime_pay
    net_pay = gross_pay - (vacation_hours * hourly_rate)

    return (gross_pay, net_pay)

# 讀取Excel表
data = pd.read_excel('employees.xlsx')

# 遍歷員工
for index, row in data.iterrows():
    print("Employee", index+1)
    hourly_rate = row['Hourly Rate']
    overtime_hours = row['Overtime Hours']
    vacation_hours = row['Vacation Hours']

    gross_pay, net_pay = calculate_pay(hourly_rate, overtime_hours, vacation_hours)

    # 輸出結果
    print("Gross pay: $", format(gross_pay, '.2f'), sep='')
    print("Net pay: $", format(net_pay, '.2f'), sep='')
    print("--------------------")