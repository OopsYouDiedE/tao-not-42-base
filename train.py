import os
import argparse
import time
import torch
import wandb
import torch.nn.functional as F
from models import TAONot42VisionModel
from load_dataset import AsyncDataBuffer, process_batch_on_gpu, CUDAPrefetcher
from utils import compute_physics_loss, extract_instances, load_yolo_backbone_weights, freeze_backbone, setup_finetune_mode
from vis import save_visualization

def train_model(args):
    device = torch.device(args.device)
    if device.type == 'cuda': torch.backends.cudnn.benchmark = True
    
    model = TAONot42VisionModel(base_channels=48, hidden_channels=768).to(device)
    if getattr(args, 'compile_model', False) and hasattr(torch, 'compile'):
        try:
            model.segmenter = torch.compile(model.segmenter, mode='reduce-overhead')
            model.depth_decoder = torch.compile(model.depth_decoder, mode='reduce-overhead')
            print("🚀 torch.compile 成功开启！")
        except Exception as e:
            print(f"⚠️ torch.compile 开启失败 (将使用正常模式): {e}")

    if args.yolo_weights:
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
            import google.colab
            from google.colab import userdata
            wandb_key = userdata.get('WANDB_API_KEY')
            if wandb_key:
                wandb.login(key=wandb_key)
        except Exception:
            pass
        wandb.init(project=args.wandb_project, config=vars(args))
        
    model.train()
    mode = "supervised"
    print(f"\n🚀 开始 TAO-NOT-42 V12 训练 (Device: {device}, Mode: {mode})")
    
    global_step = 0
    start_time = time.time()
    
    best_loss = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(1, args.epochs + 1):
        if args.finetune_after_epoch and epoch > args.finetune_after_epoch and mode == "supervised":
            mode = "self_supervised"
            setup_finetune_mode(model)
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr * 0.1)
            if scaler is not None:
                scaler = torch.amp.GradScaler(device.type)
        
        print(f"\n" + "="*40)
        print(f"🌟 Epoch {epoch}/{args.epochs} [Mode: {mode}]")
        print("="*40)
        
        epoch_loss_sum = 0.0
        
        for step_in_epoch in range(args.steps_per_epoch):
            batch = prefetcher.next()
            
            videos = batch["video"]
            b, t, c, h, w = videos.shape
            state = None
            
            for chunk_start in range(0, t, args.seq_len):
                chunk_end = min(chunk_start + args.seq_len, t)
                optimizer.zero_grad(set_to_none=True)
                total_seq_loss = 0
                loss_dict_acc = {k: 0.0 for k in ["Obj", "Box", "Mask", "Depth", "Photo", "Ego", "Flow", "Anom", "Gate", "Cls"]}
                
                for step in range(chunk_start, chunk_end):
                    x_t = videos[:, step]
                    x_next = videos[:, step+1] if step+1 < t else x_t
                    
                    time_t = torch.full((b,), step * 0.1, device=device)
                    dt_t = torch.full((b,), 1.0 / 24.0 if step > 0 else 0.0, device=device)
                    target_t = {k: v[:, step] for k, v in batch.items() if k != "video"}
                    if step > 0:
                        target_t["flow_target"] = batch["flow"][:, step - 1]
                    
                    if step + 1 < t:
                        target_t["cam_pos_next"] = batch["cam_pos"][:, step+1]
                        target_t["cam_quat_next"] = batch["cam_quat"][:, step+1]
                    else:
                        target_t["cam_pos_next"] = batch["cam_pos"][:, step]
                        target_t["cam_quat_next"] = batch["cam_quat"][:, step]
                    target_t["cam_pos_t"] = batch["cam_pos"][:, step]
                    target_t["cam_quat_t"] = batch["cam_quat"][:, step]
                    
                    with torch.autocast(device_type=device.type, enabled=(scaler is not None)):
                        out = model(x_t, dt_t, global_step, state)
                        state = out["next_state"]
                        loss, loss_dict, warped_img = compute_physics_loss(out, target_t, x_t, x_next, mode=mode, step=global_step)
                        
                    total_seq_loss += loss
                    for k in loss_dict_acc: loss_dict_acc[k] = loss_dict_acc[k] + loss_dict[k]
                    
                    if (global_step + 1) % args.vis_interval == 0 and step == chunk_end - 1:
                        filepath = save_visualization(x_t, target_t, out, global_step + 1, warped_img)
                        if args.use_wandb and filepath:
                            wandb.log({"Visualization": wandb.Image(filepath)}, step=global_step)
                            
                chunk_steps = chunk_end - chunk_start
                total_seq_loss = total_seq_loss / chunk_steps
                
                if scaler is not None:
                    scaler.scale(total_seq_loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_seq_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                
                state = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in state.items()}
                
                
                if global_step == 500 and mode == "supervised":
                    print(f"\n🔓 [Step 500] 顿悟时刻：解冻分类 Prompt，开启运动定性！")
                    if hasattr(model.segmenter.model[-1], 'class_prompts'):
                        model.segmenter.model[-1].class_prompts.requires_grad = True

                if global_step == args.unfreeze_step_1 and mode == "supervised":
                    print(f"\n🔓 [Step {args.unfreeze_step_1}] 解冻 Stage 5 高层语义...")
                    for name, param in model.segmenter.named_parameters():
                        if "stage5" in name: param.requires_grad = True
                elif global_step == args.unfreeze_step_2 and mode == "supervised":
                    print(f"\n🔓 [Step {args.unfreeze_step_2}] 解冻 Stage 4 中层特征...")
                    for name, param in model.segmenter.named_parameters():
                        if "stage4" in name: param.requires_grad = True
                        
                global_step += 1
                epoch_loss_sum += total_seq_loss.item()
                
                if global_step % 10 == 0:
                    elapsed = time.time() - start_time
                    tot_val = total_seq_loss.item()
                    cs = chunk_steps
                    
                    def get_val(k): return loss_dict_acc[k].item()/cs if isinstance(loss_dict_acc[k], torch.Tensor) else loss_dict_acc[k]/cs
                    
                    print(f"[{elapsed:.1f}s] E{epoch} S{global_step} [{mode[:3]}] | Tot:{tot_val:.4f} | "
                          f"Obj:{get_val('Obj'):.2f} Bx:{get_val('Box'):.2f} "
                          f"Msk:{get_val('Mask'):.2f} Dep:{get_val('Depth'):.2f} "
                          f"Pht:{get_val('Photo'):.2f} Ego:{get_val('Ego'):.2f} "
                          f"Flw:{get_val('Flow'):.2f} Ano:{get_val('Anom'):.2f} Cls:{get_val('Cls'):.2f}")
                    
                    if args.use_wandb:
                        log_dict = {f"Loss/{k}": get_val(k) for k in loss_dict_acc}
                        log_dict.update({
                            "Loss/Total": tot_val,
                            "System/Step": global_step,
                            "System/Epoch": epoch,
                            "System/Mode": 0 if mode == "supervised" else 1,
                            "System/Buffer_Size": len(buffer.buffer)
                        })
                        wandb.log(log_dict, step=global_step)
        
        # --- Epoch 结束 ---
        avg_epoch_loss = epoch_loss_sum / args.steps_per_epoch
        print(f"\n✅ Epoch {epoch} 结束 | 平均 Loss: {avg_epoch_loss:.4f} | Mode: {mode}")
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
            print(f"\n🛑 早停触发！")
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
            import google.colab
            from google.colab import userdata
            wandb_key = userdata.get('WANDB_API_KEY')
            if wandb_key:
                wandb.login(key=wandb_key)
        except Exception:
            pass
        wandb.init(project=args.wandb_project, config=vars(args))
        
    model.train()
    mode = "supervised"
    print(f"\n🚀 开始 TAO-NOT-42 V12 训练 (Device: {device}, Mode: {mode})")
    
    global_step = 0
    start_time = time.time()
    
    best_loss = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(1, args.epochs + 1):
        if args.finetune_after_epoch and epoch > args.finetune_after_epoch and mode == "supervised":
            mode = "self_supervised"
            setup_finetune_mode(model)
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr * 0.1)
            if scaler is not None:
                scaler = torch.amp.GradScaler(device.type)
        
        print(f"\n" + "="*40)
        print(f"🌟 Epoch {epoch}/{args.epochs} [Mode: {mode}]")
        print("="*40)
        
        epoch_loss_sum = 0.0
        
            for chunk_start in range(0, t, args.seq_len):
                chunk_end = min(chunk_start + args.seq_len, t)
                optimizer.zero_grad(set_to_none=True)
                total_seq_loss = 0
                loss_dict_acc = {k: 0.0 for k in ["Obj", "Box", "Mask", "Depth", "Photo", "Ego", "Flow", "Anom", "Gate", "Cls"]}
                
                for step in range(chunk_start, chunk_end):
                    x_t = videos[:, step]
                    x_next = videos[:, step+1] if step+1 < t else x_t
                    
                    time_t = torch.full((b,), step * 0.1, device=device)
                    dt_t = torch.full((b,), 1.0 / 24.0 if step > 0 else 0.0, device=device)
                    target_t = {k: v[:, step] for k, v in batch.items() if k != "video"}
                    if step > 0:
                        target_t["flow_target"] = batch["flow"][:, step - 1]
                    
                    if step + 1 < t:
                        target_t["cam_pos_next"] = batch["cam_pos"][:, step+1]
                        target_t["cam_quat_next"] = batch["cam_quat"][:, step+1]
                    else:
                        target_t["cam_pos_next"] = batch["cam_pos"][:, step]
                        target_t["cam_quat_next"] = batch["cam_quat"][:, step]
                    target_t["cam_pos_t"] = batch["cam_pos"][:, step]
                    target_t["cam_quat_t"] = batch["cam_quat"][:, step]
                    
                    with torch.autocast(device_type=device.type, enabled=(scaler is not None)):
                        out = model(x_t, dt_t, global_step, state)
                        state = out["next_state"]
                        loss, loss_dict, warped_img = compute_physics_loss(out, target_t, x_t, x_next, mode=mode, step=global_step)
                        
                    total_seq_loss += loss
                    for k in loss_dict_acc: loss_dict_acc[k] = loss_dict_acc[k] + loss_dict.get(k, 0.0)
                    
                    if (global_step + 1) % args.vis_interval == 0 and step == chunk_end - 1:
                        filepath = save_visualization(x_t, target_t, out, global_step + 1, warped_img)
                        if args.use_wandb and filepath:
                            wandb.log({"Visualization": wandb.Image(filepath)}, step=global_step)
                            
                chunk_steps = chunk_end - chunk_start
                total_seq_loss = total_seq_loss / chunk_steps
                
                if scaler is not None:
                    scaler.scale(total_seq_loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_seq_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                
                state = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in state.items()}
                
                if global_step == 500 and mode == "supervised":
                    print(f"\n🔓 [Step 500] 顿悟时刻：解冻分类 Prompt，开启运动定性！")
                    if hasattr(model.segmenter.model[-1], 'class_prompts'):
                        model.segmenter.model[-1].class_prompts.requires_grad = True
                
                if global_step == args.unfreeze_step_1 and mode == "supervised":
                    print(f"\n🔓 [Step {global_step}] 解冻 Stage 5 高层语义...")
                    for name, param in model.segmenter.named_parameters():
                        if "stage5" in name: param.requires_grad = True
                elif global_step == args.unfreeze_step_2 and mode == "supervised":
                    print(f"\n🔓 [Step {global_step}] 解冻 Stage 4 中层特征...")
                    for name, param in model.segmenter.named_parameters():
                        if "stage4" in name: param.requires_grad = True
                        
                global_step += 1
                epoch_loss_sum += total_seq_loss.item()
                
                if global_step % 10 == 0:
                    elapsed = time.time() - start_time
                    tot_val = total_seq_loss.item()
                    cs = chunk_steps
                    def get_val(k): return loss_dict_acc[k].item()/cs if isinstance(loss_dict_acc[k], torch.Tensor) else loss_dict_acc[k]/cs
                    print(f"[{elapsed:.1f}s] E{epoch} S{global_step} [{mode[:3]}] | Tot:{tot_val:.4f} | "
                          f"Obj:{get_val('Obj'):.2f} Bx:{get_val('Box'):.2f} "
                          f"Msk:{get_val('Mask'):.2f} Dep:{get_val('Depth'):.2f} "
                          f"Pht:{get_val('Photo'):.2f} Ego:{get_val('Ego'):.2f} "
                          f"Flw:{get_val('Flow'):.2f} Ano:{get_val('Anom'):.2f} Cls:{get_val('Cls'):.2f}")
                    if args.use_wandb:
                        log_dict = {f"Loss/{k}": get_val(k) for k in loss_dict_acc}
                        log_dict.update({"Loss/Total": tot_val, "System/Step": global_step, "System/Epoch": epoch, "System/Mode": 0 if mode == "supervised" else 1, "System/Buffer_Size": len(buffer.buffer)})
                        wandb.log(log_dict, step=global_step)
                        
        # --- Epoch 结束 ---
        avg_epoch_loss = epoch_loss_sum / args.steps_per_epoch
        print(f"\n✅ Epoch {epoch} 结束 | 平均 Loss: {avg_epoch_loss:.4f} | Mode: {mode}")
                        if "stage5" in name: param.requires_grad = True
                elif global_step == args.unfreeze_step_2 and mode == "supervised":
                    print(f"\n🔓 [Step {global_step}] 解冻 Stage 4 中层特征...")
                    for name, param in model.segmenter.named_parameters():
                        if "stage4" in name: param.requires_grad = True
                        
                global_step += 1
                epoch_loss_sum += total_seq_loss.item()
                
                if global_step % 10 == 0:
                    elapsed = time.time() - start_time
                    tot_val = total_seq_loss.item()
                    cs = chunk_steps
                    
                    def get_val(k): return loss_dict_acc[k].item()/cs if isinstance(loss_dict_acc[k], torch.Tensor) else loss_dict_acc[k]/cs
                    
                    print(f"[{elapsed:.1f}s] E{epoch} S{global_step} [{mode[:3]}] | Tot:{tot_val:.4f} | "
                          f"Obj:{get_val('Obj'):.2f} Bx:{get_val('Box'):.2f} "
                          f"Msk:{get_val('Mask'):.2f} Dep:{get_val('Depth'):.2f} "
                          f"Pht:{get_val('Photo'):.2f} Ego:{get_val('Ego'):.2f} "
                          f"Flw:{get_val('Flow'):.2f} Ano:{get_val('Anom'):.2f} Cls:{get_val('Cls'):.2f}")
                    
                    if args.use_wandb:
                        log_dict = {f"Loss/{k}": get_val(k) for k in loss_dict_acc}
                        log_dict.update({
                            "Loss/Total": tot_val,
                            "System/Step": global_step,
                            "System/Epoch": epoch,
                            "System/Mode": 0 if mode == "supervised" else 1,
                if global_step % 10 == 0:
                    elapsed = time.time() - start_time
                    tot_val = total_seq_loss.item()
                    cs = chunk_steps
                    
                    def get_val(k): return loss_dict_acc[k].item()/cs if isinstance(loss_dict_acc[k], torch.Tensor) else loss_dict_acc[k]/cs
                    
                    print(f"[{elapsed:.1f}s] E{epoch} S{global_step} [{mode[:3]}] | Tot:{tot_val:.4f} | "
                          f"Obj:{get_val('Obj'):.2f} Bx:{get_val('Box'):.2f} "
                          f"Msk:{get_val('Mask'):.2f} Dep:{get_val('Depth'):.2f} "
                          f"Pht:{get_val('Photo'):.2f} Ego:{get_val('Ego'):.2f} "
                          f"Flw:{get_val('Flow'):.2f} Ano:{get_val('Anom'):.2f} Cls:{get_val('Cls'):.2f}")
                    
                    if args.use_wandb:
                        log_dict = {f"Loss/{k}": get_val(k) for k in loss_dict_acc}
                        log_dict.update({
                            "Loss/Total": tot_val,
                            "System/Step": global_step,
                            "System/Epoch": epoch,
                            "System/Mode": 0 if mode == "supervised" else 1,
                            "System/Buffer_Size": len(buffer.buffer)
                        })
                        wandb.log(log_dict, step=global_step)
        
        # --- Epoch 结束 ---
        avg_epoch_loss = epoch_loss_sum / args.steps_per_epoch
        print(f"\n✅ Epoch {epoch} 结束 | 平均 Loss: {avg_epoch_loss:.4f} | Mode: {mode}")
        
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
            print(f"\n🛑 早停触发！")
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
    parser.add_argument('--freeze', action='store_true', default=True)
    parser.add_argument('--use_wandb', action='store_true', default=True)
    parser.add_argument('--wandb_project', type=str, default='TAO-NOT-42')
    parser.add_argument('--finetune_after_epoch', type=int, default=None)

    args = parser.parse_args()
    print('====== TAO-NOT-42 V12 配置 ======')
    for k, v in vars(args).items(): print(f"{k}: {v}")
    
    if args.mode == 'train':
        train_model(args)
