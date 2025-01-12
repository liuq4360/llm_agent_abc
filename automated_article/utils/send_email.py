import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr
import sys
sys.path.append('./')
import os
from dotenv_vault import load_dotenv  # pip install --upgrade python-dotenv-vault
load_dotenv()  # https://vault.dotenv.org/ui/ui1

SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_NAME = os.getenv("SENDER_NAME")
SENDER_PASSWD = os.getenv("SENDER_PASSWD")


# 函数：发送邮件
def send_email(receiver_email, subject, content):

    try:
        # 构建邮件
        msg = MIMEText(content, "plain", "utf-8")
        msg["From"] = formataddr((SENDER_NAME, SENDER_EMAIL))
        msg["To"] = receiver_email
        msg["Subject"] = subject

        # 使用 SMTP 发送邮件（这里以 Gmail 为例，可根据实际服务修改）
        smtp_server = "smtp.qq.com"  # 替换为你的邮件服务的 SMTP 服务器
        smtp_port = 465  # 替换为对应端口

        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.set_debuglevel(1)  # 打印调试信息
            server.login(SENDER_EMAIL, SENDER_PASSWD)
            server.sendmail(SENDER_EMAIL, receiver_email, msg.as_string())
        print(f"邮件已发送至 {receiver_email}")
    except Exception as e:
        print(f"发送邮件失败: {e}")
