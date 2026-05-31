import pandas as pd
import matplotlib.pyplot as plt
import io

def plot_results(csv_file):
    df = pd.read_csv(csv_file)
    
    # 提取 Baseline 数据
    clean_p99 = df[df['Type'] == 'Clean']['Latency_P99_us'].values[0]
    full_p99 = df[df['Type'] == 'Full']['Latency_P99_us'].values[0]
    full_bw = df[df['Type'] == 'Full']['Bandwidth_MBps'].values[0]
    
    # 提取 Sliced 数据
    sliced_df = df[df['Type'] == 'Sliced'].sort_values('Slice_KB')
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # === 绘制 P99 Latency (左轴) ===
    color = 'tab:red'
    ax1.set_xlabel('Slice Size (KB) [Log Scale]')
    ax1.set_ylabel('Victim P99 Latency (us) [Log Scale]', color=color)
    
    # 绘制 Sliced 曲线
    ax1.plot(sliced_df['Slice_KB'], sliced_df['Latency_P99_us'], marker='o', color=color, label='Ours (Sliced)')
    
    # 绘制 Baseline 参考线
    ax1.axhline(y=full_p99, color='black', linestyle='--', label=f'Full Contention ({full_p99:.0f} us)')
    ax1.axhline(y=clean_p99, color='green', linestyle=':', label=f'Clean Baseline ({clean_p99:.0f} us)')
    
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, which="both", ls="-", alpha=0.3)

    # === 绘制 Bandwidth (右轴) ===
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('PCIe Bandwidth (MB/s)', color=color)
    
    # 绘制 Sliced 带宽曲线
    ax2.plot(sliced_df['Slice_KB'], sliced_df['Bandwidth_MBps'], marker='x', linestyle='-.', color=color, label='Bandwidth')
    
    # 绘制 Full 带宽参考线
    ax2.axhline(y=full_bw, color='blue', linestyle='--', alpha=0.5, label=f'Peak Bandwidth ({full_bw/1000:.1f} GB/s)')
    
    ax2.tick_params(axis='y', labelcolor=color)
    # 设置带宽轴范围，从 0 开始以便观察下降
    ax2.set_ylim(0, full_bw * 1.2)

    # === 图例与标题 ===
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)

    # plt.title('Trade-off: Preemption Latency vs PCIe Bandwidth')
    plt.tight_layout()
    plt.savefig('latency_vs_slice_size.png')
    print("Plot saved to latency_vs_slice_size.png")

if __name__ == "__main__":
    # 如果你没有实际运行 C++，可以用下面的假数据测试绘图逻辑
    # 实际使用时，请注释掉下面这行，改用 pd.read_csv('results.csv')
    try:
        plot_results('results.csv')
    except FileNotFoundError:
        print("results.csv not found. Run ./pcie_sweep > results.csv first.")