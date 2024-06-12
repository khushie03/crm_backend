import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.utils import formataddr
import os

EMAIL_SERVER = "smtp.gmail.com"
PORT = 465
SENDER_EMAIL = "khushi2003p@gmail.com"
PASSWORD_EMAIL = "oora cvcr sjjd upgb"

def send_message(name, receiver_email):
    msg = MIMEMultipart()
    msg["Subject"] = "Missed You!"
    msg["From"] = formataddr(("CRM Team", SENDER_EMAIL))
    msg["To"] = receiver_email
    msg["BCC"] = SENDER_EMAIL

    body = f"""
    Hey there, {name}!

    Guess what? The stars have aligned, and we've noticed that it's been a little while since you last shopped with us. We've really missed your sparkling presence in our online aisles! ✨

    But fear not, because your favorite Insight Sync team is here to welcome you back with open arms and a galaxy of fantastic deals! 🚀

    Life gets busy, we totally get it. But we're thrilled to see you back in orbit with us. To celebrate your return, we've conjured up something special just for you:

    🎁 A cosmic 20% discount on your next purchase! 🎁

    Use code: COSMIC20 at checkout to blast off into savings! This discount is our way of saying thank you for being an interstellar part of the [Your Store Name] family.

    Whether you're searching for stellar new styles, cosmic home decor, or out-of-this-world gadgets, we've got everything you need to make your universe a little brighter. 🌌

    So what are you waiting for? Dive back into our celestial collection and discover galaxies of goodness waiting just for you!

    And remember, if you ever need a guiding star or have any questions, our stellar customer support team is here to assist you every step of the way.

    Welcome back to the cosmos, {name}! We can't wait to embark on this cosmic journey with you once again.
    """
    msg.attach(MIMEText(body, "plain"))

    image_path = os.path.join(os.path.dirname(__file__), 'static', 'Untitled design (1).png')

    try:
        with open(image_path, "rb") as img_file:
            img_data = img_file.read()
            img_mime = MIMEImage(img_data)
            img_mime.add_header("Content-Disposition", "attachment", filename="WelcomeBack.png")
            msg.attach(img_mime)
    except Exception as e:
        print(f"Error opening image file: {e}")

    try:
        with smtplib.SMTP_SSL(EMAIL_SERVER, PORT) as server:
            server.login(SENDER_EMAIL, PASSWORD_EMAIL)
            server.sendmail(SENDER_EMAIL, receiver_email, msg.as_string())
            print(f"Email sent to {receiver_email}")
    except Exception as e:
        print(f"Error sending email: {e}")
