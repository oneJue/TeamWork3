# Email Delivery System

本模块负责每日将推荐结果以可读格式推送到用户邮箱。

## 相关技术
- 邮件发送工具：
  - SMTP（Python smtplib / yagmail）
  - 第三方服务（如 Mailgun, SendGrid）
- 邮件内容格式：
  - Markdown 转 HTML
  - 附带原文链接与摘要

## 实用工具
- Yagmail: https://github.com/kootenpv/yagmail
- Jinja2 模板用于生成 HTML 邮件内容
