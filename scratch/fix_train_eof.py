import os

with open('train.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    if i == 391:
        new_lines.append(line)
        break
    new_lines.append(line)

tail = """        
        epoch_ckpt_path = args.checkpoint.replace(".pth", f"_epoch_{epoch}.pth")
        torch.save(model.state_dict(), epoch_ckpt_path)
        print(f"💾 已保存: {epoch_ckpt_path}")
        
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            epochs_without_improvement = 0
            best_ckpt_path = args.checkpoint.replace(".pth", "_best.pth")
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"🌟 最佳模型 (Loss: {best_loss:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"⚠️ 未优化 ({epochs_without_improvement}/{args.early_stop_patience})")
            
        if epochs_without_improvement >= args.early_stop_patience:
            print(f"\\n🛑 早停触发！")
            break
            
    print(f"🎉 训练完成。最佳: {args.checkpoint.replace('.pth', '_best.pth')}")

# =====================================================================
# Colab / GPU 训练参数配置
# =====================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TAO-NOT-42 V12 Training')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--seq_len', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--max_buffer_size', type=int, default=64)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--vis_interval', type=int, default=100)
    parser.add_argument('--compile_model', action='store_true', default=False)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--steps_per_epoch', type=int, default=1000)
    parser.add_argument('--early_stop_patience', type=int, default=3)
    parser.add_argument('--unfreeze_step_1', type=int, default=200)
    parser.add_argument('--unfreeze_step_2', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--checkpoint', type=str, default='tao_not_42_weights.pth')
    parser.add_argument('--yolo_weights', type=str, default='yolo11s-seg.pt')
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--freeze', action='store_true', default=False)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = TAONot42Base(img_size=args.img_size).to(device)

    if args.compile_model:
        try:
            model = torch.compile(model)
            print("🚀 torch.compile 开启成功")
        except Exception as e:
            print(f"⚠️ torch.compile 开启失败 (将使用正常模式): {e}")

    if args.yolo_weights:
        if not os.path.exists(args.yolo_weights):
            print(f"🌍 正在自动下载 YOLO11 预训练权重 {args.yolo_weights} ...")
            import urllib.request
            try:
                urllib.request.urlretrieve(f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{args.yolo_weights}", args.yolo_weights)
            except Exception as e:
                print(f"下载失败: {e}")
        load_yolo_backbone_weights(model, args.yolo_weights)
        if args.freeze: freeze_backbone(model)
            
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler(device.type) if device.type == 'cuda' else None
    
    buffer = AsyncDataBuffer(
        split='train', 
        max_buffer_size=args.max_buffer_size, 
        batch_size=args.batch_size, 
        max_samples=args.max_samples
    )
    prefetcher = CUDAPrefetcher(buffer, device, target_size=args.img_size)
    
    if args.use_wandb:
        try:
            wandb.init(project="tao-not-42-v12", config=vars(args))
        except:
            print("⚠️ wandb 初始化失败，将禁用。")
            args.use_wandb = False
            
    try:
        train_model(args)
    except KeyboardInterrupt:
        print("\\n🛑 训练被用户中断。")
    finally:
        buffer.stop()
"""

new_lines.append(tail)

with open('train.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
