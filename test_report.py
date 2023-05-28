import unittest
from XTestRunner import HTMLTestRunner
from test import Test_Showcv

# 创建测试套件
test_suite = unittest.TestSuite()
test_suite.addTest(unittest.makeSuite(Test_Showcv))

report_file = r'E:/test_report.html'

# 打开测试报告文件
with open(report_file, 'wb') as f:
    # 创建HTMLTestRunner运行器
    runner = HTMLTestRunner(stream=f, title='Test Report', description='Test Results')

    # 运行测试套件并生成测试报告
    runner.run(test_suite)
