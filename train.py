import wandb

try:
    wandb.login(key=c26962936685574f7d15210d5e111d36a0ca50ec)
except:
    print('To use your W&B account, go to Add-ons -> Secrets and provide your W&B access token. Use the Label name as wandb_api. \nGet your W&B access token from here: https://wandb.ai/authorize')
