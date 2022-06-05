import os
import discord
from decouple import config
from discord.ext import commands
# from dotenv import load_dotenv

# load_dotenv()

# local
# bot_token = os.getenv('TOKEN')

# heroku
bot_token = config('TOKEN')

# client = discord.Client()
client = commands.Bot("$")

# eventos
@client.event
async def on_ready():
    print('Logado, {0.user}'.format(client))

@client.event
async def on_message(message):
    if message.author == client.user:
      return
    
    # id_canal = 983118116290695238
    # canal = client.get_channel(id_canal)
    # await canal.send("Mensagem em canal especifico")

    if message.content.startswith('$teste'):
      await message.channel.send('testando 1,2,3...')
      return
    # return
    if message.attachments:
        imagem = message.attachments[0].url
        predicao = '...'
        mensagem = 'Sua predição é: ' + predicao
        await message.channel.send(mensagem)

    await client.process_commands(message)

# comandos
@client.command(name="oi")
async def send_hello(ctx):
    response = 'Olá, ' + ctx.author.name 
    await ctx.channel.send(response)

client.run(bot_token)
