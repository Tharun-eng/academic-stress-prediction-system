from firebase_admin import messaging

from firebase_config import *


def send_notification(title, body, token):

    message = messaging.Message(

        notification=messaging.Notification(
            title=title,
            body=body
        ),

        token=token

    )

    response = messaging.send(message)

    print("Notification Sent Successfully")

    print(response)