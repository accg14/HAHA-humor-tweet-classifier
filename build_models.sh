echo "Computing gru_gru models...";
sh build_gru_gru.sh;
echo "Computing gru_lstm models...";
sh build_gru_lstm.sh;
echo "Computing lstm_gru models...";
sh build_lstm_gru.sh;
echo "Computing lstm_lstm models...";
sh build_lstm_lstm.sh
