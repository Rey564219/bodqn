SHELL := powershell.exe
.SHELLFLAGS := -NoProfile -Command

PY ?= python
PAIR ?= USDJPY
DATA ?= data/$(PAIR)_M1.csv
ROWS ?=
TH_LONG ?= 0.55
TH_SHORT ?= 0.55

.PHONY: help train backtest live-fx live-crypto smoke

help:
	@Write-Host "Available targets:" -ForegroundColor Cyan
	@Write-Host "  make train                     # train_dqn.py を実行" 
	@Write-Host "  make backtest                  # regime_backtest.py を実行" 
	@Write-Host "  make backtest PAIR=EURUSD      # 通貨ペアを変更" 
	@Write-Host "  make backtest DATA=data/EURUSD_M1.csv ROWS=20000 TH_LONG=0.56 TH_SHORT=0.55"
	@Write-Host "  make live-fx                   # cTrader(Axiory) bot を起動（環境変数必須）"
	@Write-Host "  make live-crypto               # MEXC futures bot を起動（環境変数必須）"
	@Write-Host "  make smoke                     # 構文チェック"

train:
	$(PY) train_dqn.py

backtest:
	$(PY) regime_backtest.py --pair $(PAIR) --data $(DATA) --th-long $(TH_LONG) --th-short $(TH_SHORT) $(if $(ROWS),--rows $(ROWS),)

live-fx:
	$(PY) bot_fx_axiory.py

live-crypto:
	$(PY) bot_crypto_binance.py

smoke:
	$(PY) -m py_compile train_dqn.py regime_backtest.py regime_executor.py trade_core.py shared_features.py bot_fx_axiory.py bot_crypto_binance.py
