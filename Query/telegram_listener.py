from telethon import TelegramClient, events
from query import process_query
import asyncio
from llm import llm
import os

api_id = 24612008
api_hash = '563023b41e675ac0934415912c0f2fe7'
phone_number = '+6593873799'  
TARGET_CHANNEL = 'https://t.me/+n4ryVexqsAdhOGU9'

client = TelegramClient('papermatch_session', api_id, api_hash)

async def send_document_to_telegram(file_path):
    """
    Sends a document (PDF) to the Telegram channel and cleans up .json files afterward.
    """
    try:
        await client.send_file(TARGET_CHANNEL, file_path, caption="📄 Here is your PDF report.")
        print("✅ PDF sent to Telegram channel.")
    except Exception as e:
        print(f"❌ Failed to send PDF: {e}")
    finally:
        # Cleanup JSON files in eval_logs
        eval_dir = "eval_logs"
        for f in os.listdir(eval_dir):
            if f.endswith(".json"):
                try:
                    file_path = os.path.join(eval_dir, f)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        print(f"🧹 Deleted {file_path}")
                except Exception as cleanup_err:
                    print(f"❌ Error deleting file {f}: {cleanup_err}")

async def send_to_telegram_channel(message: str):
    """
    Sends a text message to the configured Telegram channel.
    """
    try:
        await client.send_message(TARGET_CHANNEL, message)
        print("✅ Sent result to Telegram channel.")
    except Exception as e:
        print(f"❌ Failed to send message to channel: {e}")

@client.on(events.NewMessage)
async def handle_message(event):
    text = event.raw_text.strip()
    user_id = event.sender_id
    chat = await event.get_chat()

    if text.lower().startswith('/ask'):
        query = text[4:].strip()

        print("----- Incoming Message -----")
        print(f"Query: {query}")
        print(f"User ID: {user_id}")
        print(f"Chat Type: {chat.__class__.__name__}")
        print(f"Chat ID: {event.chat_id}")
        print(f"Chat Title: {getattr(chat, 'title', None)}")
        print(f"Username: {getattr(chat, 'username', None)}")

        # Acknowledge user
        user = await client.get_entity(user_id)
        await event.reply(f"✅ {user.first_name}, your query has been received and is processing.")

        # Store the current event loop to pass to background callbacks
        loop = asyncio.get_running_loop()

        # Callback functions to be run in main loop
        def result_callback(message):
            asyncio.run_coroutine_threadsafe(send_to_telegram_channel(message), loop)

        def report_callback(path):
            asyncio.run_coroutine_threadsafe(send_document_to_telegram(path), loop)

        # Run processing in background thread
        def process():
            process_query(query, result_callback=result_callback)
            llm(report_callback=report_callback)

        await loop.run_in_executor(None, process)

        await event.reply(f"✅ {user.first_name}, your query has been processed. Please review the output in the report.")
        
async def main():
    await client.start(phone=phone_number)
    print("User client is running. Listening for /ask messages...")
    await client.run_until_disconnected()

if __name__ == '__main__':
    asyncio.run(main())