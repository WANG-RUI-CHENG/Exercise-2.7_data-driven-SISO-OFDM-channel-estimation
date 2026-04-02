import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results/final_results_200epochs.csv')

plt.figure(figsize=(8, 5))
plt.semilogy(df['SNR_dB'], df['DNN_CP_200ep'], 'o-', label='DNN, CP')
plt.semilogy(df['SNR_dB'], df['LMMSE_CP'], 's-', label='LMMSE, CP')
plt.semilogy(df['SNR_dB'], df['DNN_noCP_200ep'], 'o--', label='DNN, no CP')
plt.semilogy(df['SNR_dB'], df['LMMSE_noCP'], 's--', label='LMMSE, no CP')
plt.xlabel('SNR (dB)')
plt.ylabel('MSE')
plt.title('Exercise 2.7: SISO-OFDM Channel Estimation')
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('results/figure_2_9_reproduced_200ep.png', dpi=300, bbox_inches='tight')
plt.show()
