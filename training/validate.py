from typing import Union
import os
import torch
import torch.distributed as dist
from config import Config
from utils import *
from .postprocess import process_outputs,ResultSaver
import json

def validate(
    args, tasks,model, loss_fn, val_loader, epoch, device, testing=False
) -> Union[float, dict]:
    
    model.eval()
    
    model_labels,tgts_trans_for_loss,outs_trans_for_loss, outs_trans_for_res = Config.get_model_config_(
            args.model_name,"labels","targets_transform_for_loss","outputs_transform_for_loss", "outputs_transform_for_results"
        )

    average_meters = {}
    metrics_merged = {}

    sampling_rate = val_loader.dataset.sampling_rate()
    for task in tasks:
        metrics = Metrics(
            task=task,
            metric_names=Config.get_metrics(task),
            sampling_rate=sampling_rate,
            time_threshold=args.time_threshold,
            num_samples=args.in_samples,
            device=device,
        )
        metrics_merged[f"{task}"] = metrics
        for metric in metrics.metric_names():
            average_meters[f"{task}_{metric}"] = AverageMeter(
                f"[{task.upper()}]{metric}", ":6.4f"
            )

    average_meters["loss"] = AverageMeter("Loss", ":6.4f")

    progress = ProgressMeter(
        len(val_loader),
        [m for m in average_meters.values()],
        prefix=f"{'Test' if testing else 'Val'}: [{epoch}/{args.epochs}]",
    )
    
    
    if testing and args.save_test_results and is_main_process():
        results_saver = ResultSaver(item_names=tasks)
    else:
        results_saver = None
    
    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # repetitions = 500
    # import numpy as np
    # timings=np.zeros((repetitions,1))

    with torch.no_grad():
        for step, (x, loss_targets, metrics_targets, meta_data_jsons) in enumerate(val_loader):
            # if step > 600:
            #     break

            if isinstance(x, (list, tuple)):
                x = [xi.to(device) for xi in x]
            else:
                x = x.to(device)

            if isinstance(loss_targets, (list, tuple)):
                loss_targets = [yi.to(device) for yi in loss_targets]
            else:
                loss_targets = loss_targets.to(device)
            
            # if step >= 100  and step < 600:
            #     starter.record()

            outputs = model(x)

            # if step >= 100 and step < 600:
            #     ender.record()
            #     torch.cuda.synchronize()
            #     curr_time = starter.elapsed_time(ender)
            #     timings[step-100] = curr_time

            # Loss
            outputs_for_loss = outs_trans_for_loss(outputs) if outs_trans_for_loss is not None else outputs
            loss_targets = tgts_trans_for_loss(loss_targets) if tgts_trans_for_loss is not None else loss_targets
            loss = loss_fn(outputs_for_loss, loss_targets)

            # Batch size of this step
            step_batch_size = x.size(0)

            # Reduce
            if is_dist_avail_and_initialized():
                loss = reduce_tensor(loss, "AVG")
                step_batch_size = torch.tensor(
                    step_batch_size, device=device, dtype=torch.int32
                )
                step_batch_size = reduce_tensor(step_batch_size)
                dist.barrier()
                step_batch_size = step_batch_size.item()

            # Save loss
            average_meters["loss"].update(loss.item(), step_batch_size)

            # Process outputs
            outputs_for_metrics = outs_trans_for_res(outputs) if outs_trans_for_res is not None else outputs
            results = process_outputs(args, outputs_for_metrics,model_labels,sampling_rate)

            if results_saver is not None:
                if isinstance(meta_data_jsons,torch.Tensor):
                    meta_data_jsons = meta_data_jsons.detach().cpu().tolist()
                
                meta_data_dict={k:[] for k in json.loads(meta_data_jsons[0]).keys()}
                for j in meta_data_jsons:
                    for k,v in json.loads(j).items():
                        meta_data_dict[k].append(v)
                results_saver.append(meta_data_dict,metrics_targets,results)
                
            
            for task in tasks:
                metrics = Metrics(
                    task=task,
                    metric_names=Config.get_metrics(task),
                    sampling_rate=sampling_rate,
                    time_threshold=args.time_threshold,
                    num_samples=args.in_samples,
                    device=device,
                )
                metrics.compute(
                    targets=metrics_targets[task],
                    preds=results[task],
                    reduce=is_dist_avail_and_initialized(),
                )
                for metric in metrics.metric_names():
                    average_meters[f"{task}_{metric}"].update(
                        metrics.get_metric(name=metric), step_batch_size
                    )
                metrics_merged[f"{task}"].add(metrics)


            if is_main_process() and step % args.log_step == 0:
                prg_str = progress.get_str(batch_idx=step,name = f"{args.model_name}_{'test' if testing else 'val'}")
                logger.info(prg_str)

    # mean_syn = np.sum(timings) / repetitions
    # std_syn = np.std(timings)
    # print(f'Mean step time: {mean_syn:.2f} ms, std: {std_syn}')

    if results_saver is not None:
        results_save_path = get_safe_path(os.path.join(logger.logdir(),f"test_results_{val_loader.dataset.name()}.csv"))
        results_saver.save_as_csv(results_save_path)
    
    loss_avg = average_meters["loss"].avg
    return loss_avg, metrics_merged
