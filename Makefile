SHELL := powershell.exe
.SHELLFLAGS := -NoProfile -Command

PY ?= python
VENV_DIR ?= .venv
PAIR ?= USDJPY
DATA ?= data/$(PAIR)_M1.csv
ROWS ?=
TH_LONG ?= 0.55
TH_SHORT ?= 0.55

.PHONY: help venv train backtest live-fx live-fx-check live-crypto live-crypto-check smoke clean

help:
	@Write-Host "Available targets:" -ForegroundColor Cyan
	@Write-Host "  make venv                      # .venv を作成してpip更新"
	@Write-Host "  make train                     # train_dqn.py を実行" 
	@Write-Host "  make backtest                  # regime_backtest.py を実行" 
	@Write-Host "  make backtest PAIR=EURUSD      # 通貨ペアを変更" 
	@Write-Host "  make backtest DATA=data/EURUSD_M1.csv ROWS=20000 TH_LONG=0.56 TH_SHORT=0.55"
	@Write-Host "  make live-fx                   # cTrader(Axiory) bot を起動（環境変数必須）"
	@Write-Host "  make live-fx-check             # cTrader事前確認（初期化のみ、注文なし）"
	@Write-Host "  make live-crypto               # Bitget futures bot を起動（環境変数必須）"
	@Write-Host "  make live-crypto-check         # Bitget接続確認（残高取得のみ、注文なし）"
	@Write-Host "  make smoke                     # 構文チェック"
	@Write-Host "  make clean                     # Pythonキャッシュを削除"

venv:
	@if (!(Test-Path "$(VENV_DIR)/Scripts/python.exe")) { $(PY) -m venv $(VENV_DIR) }
	@$(VENV_DIR)/Scripts/python.exe -m pip install --upgrade pip
	@Write-Host "[OK] venv ready: $(VENV_DIR)"

train:
	$(PY) train_dqn.py

backtest:
	$(PY) regime_backtest.py --pair $(PAIR) --data $(DATA) --th-long $(TH_LONG) --th-short $(TH_SHORT) $(if $(ROWS),--rows $(ROWS),)

live-fx:
	$(PY) bot_fx_axiory.py

live-fx-check:
	$(PY) -c "from bot_fx_axiory import AxioryCTraderBot, CTRADER_CLIENT_ID, CTRADER_CLIENT_SECRET, CTRADER_ACCESS_TOKEN, CTRADER_ACCOUNT_ID, CTRADER_SYMBOL; assert CTRADER_CLIENT_ID and CTRADER_CLIENT_SECRET and CTRADER_ACCESS_TOKEN and CTRADER_ACCOUNT_ID, 'Set CTRADER_CLIENT_ID, CTRADER_CLIENT_SECRET, CTRADER_ACCESS_TOKEN, CTRADER_ACCOUNT_ID'; b=AxioryCTraderBot(); print('[OK] cTrader preflight initialized'); print('[SYMBOL]:', CTRADER_SYMBOL); print('[ACCOUNT_ID]:', CTRADER_ACCOUNT_ID); print('[MODEL_PAIR]:', b.model_pair)"

live-crypto:
	$(PY) bot_crypto_bitget.py

live-crypto-check:
	$(PY) -c "from bot_crypto_bitget import BitgetFuturesClient, BASE_URL, API_KEY, API_SECRET, API_PASSPHRASE; assert API_KEY and API_SECRET and API_PASSPHRASE, 'Set BITGET_API_KEY, BITGET_API_SECRET and BITGET_API_PASSPHRASE'; c=BitgetFuturesClient(BASE_URL, API_KEY, API_SECRET, API_PASSPHRASE); data=c.get_balance(); print('[OK] Bitget account rows:', len(data)); usdt=next((x for x in data if str(x.get('marginCoin','')).upper()=='USDT'), None); print('[USDT available]:', usdt.get('available') if usdt else 'not found')"

smoke:
	$(PY) -m py_compile train_dqn.py regime_backtest.py regime_executor.py trade_core.py shared_features.py bot_fx_axiory.py bot_crypto_bitget.py

clean:
	@Get-ChildItem -Path . -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
	@Get-ChildItem -Path . -Recurse -File -Include *.pyc,*.pyo | Remove-Item -Force -ErrorAction SilentlyContinue
	@if (Test-Path "$(VENV_DIR)") { Remove-Item "$(VENV_DIR)" -Recurse -Force -ErrorAction SilentlyContinue }
