from slack import WebClient
from slack.errors import SlackApiError
from ..utils import logger


def send_message(message: str, channel: str, token: str, attachment=None, as_attachment=False, divider=False):

    client = WebClient(token=token)

    try:
        if as_attachment or divider:
            attachments = [
                {
                    "text": message,
                    "fallback": message,
                    "fields": {} if attachment == None else [
                        {"title": key, "value": value, "short": True} for key, value in attachment.items()
                    ]
                }
            ]
        else:
            attachments = []
        blocks = [{"type": "divider"}] if divider else []

        return client.api_call(api_method="chat.postMessage",
                               json={'channel': channel, "text": message, "attachments": attachments, "blocks": blocks})

    except SlackApiError as e:
        logger.error(f"Got an error: {e.response['error']}")


def divider(channel: str, token: str):
    send_message("", channel, token, divider=True)


def send_metrics(epoch: int, run_name: str, values: dict, channel: str, token: str):
    title = "*Epoch %s* %s" % (str(epoch), str(run_name))
    items = "\n".join(["- %s: %.5f" % (n, v) for n, v in values.items()])
    send_message("%s\n%s" % (title, items), channel, token)


if __name__ == "__main__":
    import os
    import json

    metrics = {
        "loss": 0.12121,
        "val_loss": 0.212121,
        "iou_score": 0.45,
        "f1_score": 0.56,
    }
    send_metrics(1, metrics, 'training', os.environ['SLACK_TOKEN'])

    # send_message('*Epoch 100*\n```%s```' % json.dumps(message), 'training', os.environ['SLACK_TOKEN'])
    # send_message('*Epoch 100*\n%s' % "\n".join(["- %s: %.5f" % (n, v) for n, v in message.items()]), 'training', os.environ['SLACK_TOKEN'])

    # send_message('Support Ticket #5', {
    #     "user_id": 1234,
    #     "subject": "Cannot create charge",
    #     "description": "This is a detailed description of that is about to happen"
    # })
