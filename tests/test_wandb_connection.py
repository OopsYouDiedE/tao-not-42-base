import os
import wandb

def main():
    print("====================================================")
    print("[测试] 开始验证本机 Weights & Biases (wandb) 连接性")
    print("====================================================")
    
    # 1. 从 .env 文件手动加载 WANDB_API API Key
    api_key = None
    env_path = ".env"
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                if "WANDB_API=" in line:
                    api_key = line.split("WANDB_API=")[1].strip()
                    break
                    
    if api_key:
        # 设置 wandb 识别的 API 密钥环境变量
        os.environ["WANDB_API_KEY"] = api_key
        print("[信息] 成功从 .env 文件读取并设置 WANDB_API_KEY")
    else:
        print("[错误] 未能在 .env 文件中查找到 WANDB_API 字段！")
        return
        
    try:
        # 2. 尝试登录 wandb
        print("正在尝试登录 Weights & Biases...")
        login_success = wandb.login()
        print(f"wandb 登录结果: {login_success}")
        
        if login_success:
            # 3. 初始化测试项目 run 并尝试上传指标
            print("正在初始化测试 wandb 运行以验证通信...")
            run = wandb.init(
                project="tao-not-42-test", 
                name="local-connection-verification",
                config={"env": "local_windows_development"}
            )
            
            wandb.log({"verification_metric": 42.0, "status": 1})
            print("[成功] 已成功向 wandb 上传验证指标 {verification_metric: 42.0}")
            
            # 结束当前 run
            run.finish()
            print("====================================================")
            
            # 创建一个用于保存连接测试结果的文件，供后续参考
            with open("vis_outputs/wandb_status.txt", "w", encoding="utf-8") as f:
                f.write("Weights & Biases connection successful!")
                
            print("[结论] 本地机器上的 Weights & Biases 通信与登录验证完全通过！")
            print("====================================================")
        else:
            print("[错误] Weights & Biases 登录未成功，请检查 API Key 的有效性。")
    except Exception as e:
        print(f"[异常] 验证 wandb 时发生错误: {e}")

if __name__ == "__main__":
    main()
