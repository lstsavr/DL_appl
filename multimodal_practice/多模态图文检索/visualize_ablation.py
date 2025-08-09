import os
import json
import matplotlib.pyplot as plt
import numpy as np

def load_ablation_results():
    """直接从各个实验目录加载消融实验结果"""
    # 预期的实验顺序
    experiment_order = [
        "基线",
        "+ ResNet101",
        "+ Mean_Max池化",
        "+ 困难负样本挖掘",
        "+ 渐进式解冻",
        "完整模型"
    ]
    
    results = {}
    
    # 尝试从每个实验目录加载结果
    for exp_name in experiment_order:
        exp_dir = os.path.join("ablation_results", exp_name.replace(" ", "_"))
        summary_file = os.path.join(exp_dir, "summary.json")
        
        if os.path.exists(summary_file):
            with open(summary_file, 'r', encoding='utf-8') as f:
                try:
                    summary = json.load(f)
                    results[exp_name] = {
                        "Mean_R@1": summary.get("best_mean_r1", 0.0),
                    }
                    print(f"加载数据: {exp_name} = {results[exp_name]['Mean_R@1']:.4f}")
                except json.JSONDecodeError:
                    print(f"警告: {summary_file} 不是有效的JSON文件")
                    continue
        else:
            print(f"警告: 找不到实验 '{exp_name}' 的结果文件 {summary_file}")
    
    # 如果存在基线结果，计算相对提升
    if "基线" in results:
        baseline = results["基线"]["Mean_R@1"]
        
        # 计算每个实验相对基线的提升
        for exp_name, metrics in results.items():
            if exp_name != "基线":
                absolute_imp = metrics["Mean_R@1"] - baseline
                relative_imp = (absolute_imp / baseline) * 100
                
                # 添加到结果字典
                metrics["absolute_improvement"] = absolute_imp
                metrics["relative_improvement_percent"] = relative_imp
    
    # 创建最终报告格式
    final_report = {
        "results": results,
    }
    
    return final_report

def create_ablation_chart(report):
    """创建消融实验图表"""
    # 准备数据
    configurations = []
    mean_r1_scores = []
    relative_improvements = []
    
    # 指定顺序
    config_order = ["基线", "+ ResNet101", "+ Mean_Max池化", 
                  "+ 困难负样本挖掘", "+ 渐进式解冻", "完整模型"]
    
    # 按指定顺序提取数据
    for config in config_order:
        if config in report["results"]:
            configurations.append(config.replace('+', '+\n'))  # 添加换行以便显示
            mean_r1_scores.append(report["results"][config]["Mean_R@1"])
            
            # 安全地获取相对改进百分比
            if config != "基线" and "relative_improvement_percent" in report["results"][config]:
                relative_improvements.append(report["results"][config]["relative_improvement_percent"])
            else:
                # 如果是基线或没有相对改进数据，添加0
                relative_improvements.append(0.0)
    
    # 创建图表目录
    os.makedirs('figures', exist_ok=True)
    
    # 创建柱状图
    plt.figure(figsize=(14, 8))
    
    # 设置中文字体支持
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'SimSun', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("警告: 无法设置中文字体，图表中的中文可能无法正确显示")
    
    bars = plt.bar(
        configurations, 
        mean_r1_scores, 
        color='skyblue', 
        edgecolor='black',
        width=0.6
    )
    
    # 获取基线值用于计算百分比
    baseline_value = mean_r1_scores[0] if mean_r1_scores else 0
    
    # 添加值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        # 在柱子顶部添加数值标签
        plt.text(
            bar.get_x() + bar.get_width() / 2, 
            height + 0.005,
            f'{height:.3f}', 
            ha='center', 
            va='bottom',
            fontsize=11, 
            fontweight='bold'
        )
        
        # 添加相对提升百分比
        if i > 0 and baseline_value > 0:  # 跳过基线并确保基线不为零
            # 手动计算相对提升
            rel_improvement = ((height - baseline_value) / baseline_value) * 100
            plt.text(
                bar.get_x() + bar.get_width() / 2, 
                height / 2,
                f'+{rel_improvement:.1f}%', 
                ha='center', 
                va='center',
                fontsize=12, 
                color='black', 
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3')
            )
    
    # 设置标题和标签
    plt.title('模型配置的消融实验结果（真实数据）', fontsize=16, pad=15)
    plt.ylabel('Mean R@1', fontsize=14, labelpad=10)
    plt.xlabel('模型配置', fontsize=14, labelpad=10)
    
    # 优化网格线和刻度
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.gca().set_axisbelow(True)  # 网格线置于图形之下
    
    # 设置坐标轴刻度字体大小
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    # 调整Y轴范围，留出顶部空间显示数字
    plt.ylim(0, max(mean_r1_scores) * 1.1)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存高质量图像
    plt.savefig('figures/real_ablation_study.png', dpi=300, bbox_inches='tight')
    print("✓ 已保存真实消融实验结果图: figures/real_ablation_study.png")
    
    # 保存SVG矢量格式
    try:
        plt.savefig('figures/real_ablation_study.svg', format='svg', bbox_inches='tight')
        print("✓ 已保存SVG矢量图: figures/real_ablation_study.svg")
    except:
        print("保存SVG格式失败，可能需要安装额外依赖")
    
    plt.close()

def generate_ablation_table(report):
    """生成消融实验表格，用于更新报告"""
    # 指定顺序
    config_order = ["基线", "+ ResNet101", "+ Mean_Max池化", 
                  "+ 困难负样本挖掘", "+ 渐进式解冻", "完整模型"]
    
    # 打印Markdown表格
    print("\n### 真实消融实验结果（用于更新报告）\n")
    print("**表2: 消融实验结果（Mean R@1）- 真实实验数据**\n")
    print("| 实验配置 | Mean R@1 | 相对基线提升 |")
    print("|---------|----------|-------------|")
    
    # 安全获取基线值
    baseline = 0
    if "基线" in report["results"]:
        baseline = report["results"]["基线"]["Mean_R@1"]
    
    for config in config_order:
        if config in report["results"]:
            mean_r1 = report["results"][config]["Mean_R@1"]
            if config == "基线" or baseline == 0:
                rel_imp = "-"
            else:
                rel_imp = f"+{(mean_r1 - baseline) / baseline * 100:.1f}%"
            
            print(f"| {config} | {mean_r1:.3f}    | {rel_imp} |")
    
    print("\n以上数据来自于实际运行的消融实验，结果保存在`ablation_results`目录。")

def main():
    """主函数"""
    print("开始生成消融实验结果可视化...")
    
    # 加载结果
    report = load_ablation_results()
    if report is None:
        return
    
    # 创建图表
    create_ablation_chart(report)
    
    # 生成表格
    generate_ablation_table(report)
    
    print("\n消融实验可视化完成！")
    
if __name__ == "__main__":
    main()
