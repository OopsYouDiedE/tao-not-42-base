import os

with open('train.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    if i == 135:
        break
    new_lines.append(line)

tail = """                
                if global_step == 500 and mode == "supervised":
                    print(f"\\n🔓 [Step 500] 顿悟时刻：解冻分类 Prompt，开启运动定性！")
                    if hasattr(model.segmenter.model[-1], 'class_prompts'):
                        model.segmenter.model[-1].class_prompts.requires_grad = True

                if global_step == args.unfreeze_step_1 and mode == "supervised":
                    print(f"\\n🔓 [Step {args.unfreeze_step_1}] 解冻 Stage 5 高层语义...")
                    for name, param in model.segmenter.named_parameters():
                        if "stage5" in name: param.requires_grad = True
                elif global_step == args.unfreeze_step_2 and mode == "supervised":
                    print(f"\\n🔓 [Step {args.unfreeze_step_2}] 解冻 Stage 4 中层特征...")
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
        print(f"\\n✅ Epoch {epoch} 结束 | 平均 Loss: {avg_epoch_loss:.4f} | Mode: {mode}")
"""

new_lines.append(tail)

# Append everything from epoch_ckpt_path onwards
for i, line in enumerate(lines):
    if i >= 138 and "epoch_ckpt_path = args.checkpoint.replace" in line:
        new_lines.extend(lines[i:])
        break

with open('train.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
