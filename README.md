# Large-Scale-AI-Training-Playbook


#### Determining the feasible batch sizes and estimating training throughput

<details><summary><em>[Click to expand]</em></summary>

<br>

-   For a given model and optimizer, there will typically be a range of batch
    sizes supported by the available hardware. The limiting factor is usually
    accelerator memory.
-   Unfortunately, it can be difficult to calculate which batch sizes will fit
    in memory without running, or at least compiling, the full training program.
-   The easiest solution is usually to run training jobs at different batch
    sizes (e.g. increasing powers of 2) for a small number of steps until one of
    the jobs exceeds the available memory.
-   For each batch size, we should train for long enough to get a reliable
    estimate of the *training throughput*

<p align="center">training throughput = (# examples processed per second)</p>

<p align="center">or, equivalently, the <em>time per step</em>.</p>

<p align="center">time per step = (batch size) / (training throughput)</p>

-   When the accelerators aren't yet saturated, if the batch size doubles, the
    training throughput should also double (or at least nearly double).
    Equivalently, the time per step should be constant (or at least nearly
    constant) as the batch size increases.
-   If this is not the case then the training pipeline has a bottleneck such as
    I/O or synchronization between compute nodes. This may be worth diagnosing
    and correcting before proceeding.
-   If the training throughput increases only up to some maximum batch size,
    then we should only consider batch sizes up to that maximum batch size, even
    if a larger batch size is supported by the hardware.
    -   All benefits of using a larger batch size assume the training throughput
        increases. If it doesn't, fix the bottleneck or use the smaller batch
        size.
    -   **Gradient accumulation** simulates a larger batch size than the
        hardware can support and therefore does not provide any throughput
        benefits. It should generally be avoided in applied work.
-   These steps may need to be repeated every time the model or optimizer is
    changed (e.g. a different model architecture may allow a larger batch size
    to fit in memory).

</details>
