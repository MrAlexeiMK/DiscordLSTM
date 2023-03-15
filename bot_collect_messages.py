import os
import getopt
import sys
import discord
import csv
import pathlib
from bot_train import *


def is_not_valid(msg):
    return '<' in msg or '>' in msg or '$' in msg or "http" in msg


def run(token, userId, channelId, limit, outputFile):
    print("[INFO] All data will be stored in '" + outputFile + "'")

    intents = discord.Intents.all()
    client = discord.Client(intents=intents)
    script_dir = pathlib.Path(__file__).parent.absolute()
    os.chdir(script_dir)

    @client.event
    async def on_ready():
        print(f'[INFO] {client.user} has connected to Discord!')

        with open(outputFile, 'a', newline='', encoding="utf-8") as file:
            writer = csv.writer(file, delimiter='$')

        for channel in client.get_all_channels():
            if channelId != None and channel.id != channelId:
                continue

            if not isinstance(channel, discord.TextChannel):
                continue
            
            if channel.is_nsfw == True:
                continue

            print("[INFO] Collecting messages from channel", channel.id)

            async for message in channel.history(limit=limit):
                async for previous_message in channel.history(limit=1, before=message):
                    if userId != None and message.author.id != userId:
                        continue

                    question = previous_message.content
                    answer = message.content

                    if is_not_valid(question) or is_not_valid(answer):
                        continue

                    try:
                        with open(outputFile, 'a', newline='', encoding="utf-8") as file:
                            writer = csv.writer(file, delimiter='$')
                            writer.writerow([question, answer])
                    except:
                        pass

    client.run(token)

def collect_messages_help():
    print('[INFO] python bot_collect_messages.py -t <discordBotToken> -u [userId] -c [channelId] --limit [messagesLimit] -o [outputfile]')
    sys.exit()

def main(argv):
    token = None
    userId = None
    outputFile = None
    channelId = None
    limit = 100000

    opts, args = getopt.getopt(argv,"ht:u:c:l:o:",["token=", "userId=", "channelId=", "limit=", "outputFile="])
    for opt, arg in opts:
        if opt == '-h':
            help()
        elif opt in ("-t", "--token"):
            token = arg
        elif opt in ("-u", "--user", "--userId"):
            userId = int(arg)
        elif opt in ("-c", "--channel", "--channelId"):
            channelId = int(arg)
        elif opt in ("-l", "--limit"):
            limit = int(arg)
        elif opt in ("-o", "--output", "--outputFile"):
            outputFile = arg

    cancelled = False
    if token == None:
        print("[ERROR] You need to provide discord-bot token")
        cancelled = True
    if userId == None and outputFile == None:
        print("[ERROR] You need to specify output file")
        cancelled = True

    if cancelled == True:
        collect_messages_help()

    if outputFile == None:
        outputFile = "data/discord_conversation_" + str(userId) + ".csv"
    
    run(token, userId, channelId, limit, outputFile)

if __name__ == "__main__":
    main(sys.argv[1:])