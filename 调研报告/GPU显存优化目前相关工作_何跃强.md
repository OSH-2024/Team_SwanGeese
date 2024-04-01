## GPU显存优化目前相关工作

当今，随着深度学习的发展，模型变得越来越大，对显存的需求也越来越多，GPU的显存急需优化方案。优化GPU显存的主要目标是最大程度地减少显存的使用量，以提高程序的性能和效率。以下我将介绍GPU优化方案的发展历程与GPU优化的优秀成果。

**2016年：** 

​		发表在MICRO16（体系结构A会）上的vDNN[1]提出了一种方案，即在训练时，把训练当前阶段不需要的数据转移到CPU上，借助CPU上的大内存暂存数据。因为DNN模型是按照层进行排列的，每一个时刻GPU其实只训练某个层，势必会有一些层的数据我们暂时不需要访问，所以这些数据便可以转移出去。在文章中，作者对CV模型进行测试，发现Feature map类型的数据占空间是最大的，所以重点是面向Feature map的数据进行优化。进一步，作者发现CV主要包括两个比较重要的层，卷积以及FC全连接。而这两个层中，卷积层的输入占用空间特别多，于是作者就把卷积层的Tensor在前传的时候转出到CPU里面，等到反传需要的时候再提起转回来。但是这种方案有缺点，即由于同步的影响，导致了严重的性能开销，且比较死板，先入为主的就对卷积做策略，随着未来模型的发展，这种方案势必不智能；

​		之后，Tianqi Chen另辟蹊径，提出了DNN模型层的重计算思想[2]。将暂时用不到的显存释放掉，然后反传用到时再通过前传得到。虽然能释放大量显存，但是计算开销是串行的，特别大。

**2018-2019年：**

​		其中一部分工作主要在vDNN的方向上进行优化，代表性的工作有DATE18的MoDNN[3]、IPDPS19的vDNN++[4]。一部分工作将转移、重计算结合到一起，例如SuperNeurons[5]，它提出了一个系统级的优化框架。还有一部分工作另辟蹊径，将压缩引入DNN的显存优化，从而开辟了新的领域，如-Gist[6]、cDMA[7]。

​		MoDNN是2018年发表在DATE（CCF-B）上的一篇基于转移的显存优化工作。他的核心发现是DNN框架中，卷积函数包括很多不同的类别，有的卷积函数空间使用多，但是性能较快（FFT、Winograd algorithm），有的不占用太多内存，但是性能相对较差（GEMM）。所以moDNN能够只能的选择卷积函数，并调整mini-batchsize来最优化系统的执行性能，并智能选择转移策略。vDNN++对GPU碎片问题，转移后的数据压缩问题进行了进一步优化。

​		SuperNeurons是2018年比较有影响力的文章，其核心思想可以归结为下面三点，从而针对不同类别的层进行不同的策略，最终一步一步释放显存。

![img](https://pic4.zhimg.com/v2-9b55923f2e094bfd5e76f22b158a1f0f_r.jpg)

然而，作为早期文章，该方案同样先入为主的对当时比较主流的几个模型进行了分析，并有使用固定思维对某些层做某些事情，不够灵活。

​		Gist[6]针对ReLU的输出进行有损+无损的压缩，释放了许多ReLU层相关的显存，节省了空间。cDMA[7]利用了ReLU输出层的稀疏特性，并在GPU中做了一个硬件core来压缩这些数据，以减少转移的数据量，最终提升性能。但是这种硬件的文章其实都是基于模拟器做，也无法为我们所用。并且压缩算法定制到硬件上，也比较单一，所以弊端还是很多的。

**2020年：**

​		在ASPLOS'20这一会议中，涌出了一批相对比较优秀的GPU显存解决工作。

​		首先是SwapAdvisor[8]。这篇工作的想法比较简单，作者认为之前的基于人工判断的转移方法并不高效（例如vDNN的只对卷积转移）。所以他认为系统可以使用传统的启发式搜索方案来进行转移策略的搜索，这里选择的是遗传算法。虽然想法简单，但是作者还是经过了大量的实验测试以及对比，最终选择了遗传算法，并且针对遗传算法的流程以及DNN的流程，将转移这个操作建模出来，并且也对非线性网络的支持比较好。

​		AutoTM[9]是ASPLOS'20的另一篇文章，这个文章的大背景就是使用暴力搜索（线性整数规划）来搜索出来合适的转移策略。他是第一个使用CPU的DRAM+NVM（非易失性内存）进行策略搜索的工作，并且开源了代码。不过对于搜索范围较大的或是异步运行的时候表现性能较差。

​		除了上述几篇转移的工作外，本年的ASPLOS会议上还有一篇GPU显存优化的工作-Capuchin[10]，而该工作就是在SuperNeurons的基础上，做的更深更好（转移+重计算）。这篇文章解决了很多GPU显存的问题。简单来说，该工作有如下三点Motivation：

<img src="https://pic2.zhimg.com/80/v2-a8b4b13cad434338e1b6c781f079484d_720w.webp" alt="img" style="zoom:80%;" />

不能先入为主的对各种层做特定操作（例如就对卷积的Tensor做转移，对Pool做重计算等等），这样势必不能达到比较高效（也不能是最高效），于是这个文章先详尽一切办法把能并行的转移并行进来，然后迭代调出最优的重计算方案，直到内存容量足够为止。本文引入公式，对不同Tensor按照下面的公式计算，从而排序进行优化：

![img](https://pic2.zhimg.com/80/v2-e3eaef411314a39a0d6bd330401d8c7d_720w.webp)

而这个公式将Tensor按照内存保存量以及重计算的时间需要进行了权重计算，简单来说，Tensor中重计算时间越少，内存保存量越大的数据，越值得我们首先考虑。

相比于之前的SuperNeurons，这个工作就无关什么层的类型，我一视同仁，只要你能并行，我就并行起来，并且选择最优的重计算方案。对比其他的仅转移方案的文章，本文引入了重计算，从而在某些情况下重计算的开销会比转移低，从而提升模型训练的性能。

**2021年：**

​		2021年在显存优化上出现了颇多更有意思的点，例如FlashNeuron[11]。这个论文一反常态，认为上述一堆文章都是把数据转移到CPU的DRAM上，但是那些文章都没有考虑过内存、CPU正在执行数据预处理操作，从而使得内存总线始终在忙碌，从而使得转移性能极差，转移性能随着CPU忙碌而变差。于是这个文章破天荒的将数据转移到SSD上，并利用了比较简单的转移方法，引入压缩从而进一步降低转移数据量。

​		该年还有一篇显存优化的文章：Sentinel来自HPCA21[12]，他算是第二篇将NVM与DRAM结合的文章，但是他的点比较微观，抓住的是转移页的时候，生命周期不同的的数据被分配到了同一个页中，造成了大量不必要的转移。所以这个文章把这个问题解决了。并且实验对比了当前最好的一系列文章。

​		再者就是ATC21的ZeRO-Offload文章[13]。这个文章发现，NLP里面有许多参数其实是为优化器所用的。与CV不同，CV是Feature map特别大，而NLP偏向parameter，而优化器内部其实是和weight的量成正比的。加上momentum以及variance大概能有3倍。所以这个文章把优化器所有参数卸载到CPU上，包括计算在内，并设计了更快的CPU优化器运算，从而做到了完美的GPU显存优化。

​		然后是2021年CLUSTER21的关于DNN+压缩的优化文章-CSWAP。这个文章利用了ReLU输出数据的稀疏特性、稀疏可变特性（随着训练的进行不同层的稀疏程度是可变的），提出了选择性压缩的架构，在转移的过程中，如果数据压缩+转移性能好，那我们就压缩，否则就不压缩。为了达成这个目的，本文利用了2个机器学习模型对一些指标进行预测，从而比较准确的挑选出合适的层进行压缩+转移，其余只转移，从而在优化显存的同时，最优化训练性能。

​		最后是《DeepCuts: A Deep Learning Optimization Framework for Versatile GPU Workloads》[14],目前绝大多数深度学习框架都是使用cuDNN原生高性能库对在GPU上进行的计算做加速优化。然而随着深度学习领域的快速发展，基于cuDNN的框架在GPU优化上并不总能给出最优解。因此论文作者提出了DeepCuts，一个专为解决深度学习应用中越来越多样的workloads而设计的深度学习框架。其不同于上述其他框架的两个重要创新点：1、直接使用GPU硬件参数；2、不实际评估每个参数组合的性能，而是通过估计性能上界去筛除理论计算结果较差的参数组合，缩小范围。

对于一个给定的深度学习模型中的计算操作，DeepCuts会搜索产生最优性能GPU kernel的参数组合，并在搜索过程中尝试算子融合，并使用具有后端架构感知的性能估计模型对“definitely-low”的参数组合进行剪枝。基于选中的参数组合，生成数据流图并进一步生成GPU kernel。最后执行这些kernel并为每个算子或融合算子选出使其性能最优的。

**总结来看，GPU显存优化的历史便在下面的表格：**

| 年份 | 会议    | 题目         | 方案        | 备注                                     |
| ---- | ------- | ------------ | ----------- | ---------------------------------------- |
| 2016 | MICRO   | VDNN         | 转移        | 第一篇转移                               |
| 2016 | arXiv   | Checkpoint   | 重计算      | 第一篇重计算                             |
| 2018 | DATE    | moDNN        | 转移        | 考虑不同卷积函数的性能                   |
| 2018 | PPoPP   | SuperNeurons | 转移+重计算 | 第一篇转移+重计算                        |
| 2018 | HPCA    | cDMA         | 转移+压缩   | 硬件对ReLU输出层压缩                     |
| 2019 | IPDPS   | vDNN++       | 转移+压缩   | 针对vDNN的性能问题进行解决               |
| 2020 | ASPLOS  | AutoTM       | 转移        | NVM+DRAM的暴力搜索                       |
| 2020 | ASPLOS  | SwapAdvisor  | 转移        | 启发式搜索+GPU碎片问题                   |
| 2020 | ASPLOS  | Capuchin     | 转移+重计算 | 转移+重计算的较为优秀的方案              |
| 2021 | FAST    | FlashNeuron  | 转移        | 引入SSD转移                              |
| 2021 | ATC     | ZeRO-Offload | 转移        | 利用CPU的计算能力                        |
| 2021 | CLUSTER | CSWAP        | 转移+压缩   | 对稀疏数据进行选择性压缩，最优化性能     |
| 2021 | HPCA    | Sentinel     | 转移        | NVM+DRAM场景，考虑细粒度Page上的数据情况 |

​		通过以上研究成果，我们可以看到，目前结合机器学习与GPU显存优化的例子较少，可见在利用机器学习进行筛选算法、数据的领域中我们可以有更多的想法与进展。

### 参考文献



[1]M. Rhu, N. Gimelshein, J. Clemons, A. Zulfiqar, and S. W. Keckler, “VDNN: Virtualized deep neural networks for scalable, memory-efficient neural network design,” in *Proceedings of the Annual International Symposium on Microarchitecture, MICRO*, 2016, vol. 2016-Decem.

[2] T. Chen, B. Xu, C. Zhang, and C. Guestrin, “Training Deep Nets with Sublinear Memory Cost,” pp. 1–12, 2016.

[3] X. Chen, D. Z. Chen, and X. S. Hu, “MoDNN: Memory optimal DNN training on GPUs,” *Proc. 2018 Des. Autom. Test Eur. Conf. Exhib. DATE 2018*, vol. 2018-Janua, pp. 13–18, 2018.

[4] S. B. Shriram, A. Garg, and P. Kulkarni, “Dynamic memory management for GPU-based training of deep neural networks,” *Proc. - 2019 IEEE 33rd Int. Parallel Distrib. Process. Symp. IPDPS 2019*, pp. 200–209, 2019.

[5] L. Wang *et al.*, “SuperNeurons: Dynamic GPU memory management for training deep neural networks,” in *Proceedings of the ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming, PPOPP*, 2018, pp. 41–53.

[6] A. Jain, A. Phanishayee, J. Mars, L. Tang, and G. Pekhimenko, “GIST: Efficient data encoding for deep neural network training,” *Proc. - Int. Symp. Comput. Archit.*, pp. 776–789, 2018.

[7] M. Rhu, M. O’Connor, N. Chatterjee, J. Pool, Y. Kwon, and S. W. Keckler, “Compressing DMA Engine: Leveraging Activation Sparsity for Training Deep Neural Networks,” *Proc. - Int. Symp. High-Performance Comput. Archit.*, vol. 2018-Febru, pp. 78–91, 2018.

[8] C. C. Huang, G. Jin, and J. Li, “SwapAdvisor: Pushing deep learning beyond the GPU memory limit via smart swapping,” in *International Conference on Architectural Support for Programming Languages and Operating Systems - ASPLOS*, 2020, pp. 1341–1355.

[9]M. Hildebrand, J. Khan, S. Trika, J. Lowe-Power, and V. Akella, “AutOTM: Automatic tensor movement in heterogeneous memory systems using integer linear programming,” in *International Conference on Architectural Support for Programming Languages and Operating Systems - ASPLOS*, 2020, pp. 875–890.

[10] X. Peng *et al.*, “Capuchin: Tensor-based GPU memory management for deep learning,” in *International Conference on Architectural Support for Programming Languages and Operating Systems - ASPLOS*, 2020, pp. 891–905.

[11] J. Bae *et al.*, “FlashNeuron : SSD-Enabled Large-Batch Training of Very Deep Neural Networks This paper is included in the Proceedings of the 19th USENIX Conference on File and Storage Technologies .,” 2021.

[12] J. Ren, J. Luo, K. Wu, M. Zhang, H. Jeon, and D. Li, “Sentinel : Efficient Tensor Migration and Allocation on Heterogeneous Memory Systems for Deep Learning,” pp. 598–611, 2021.

[13] J. Ren *et al.*, “ZeRO-Offload : Democratizing Billion-Scale Model Training This paper is included in the Proceedings of the,” 2021.

[14]Wookeun Jung, Thanh Tuan Dao, Jaejin Lee,"DeepCuts: A Deep Learning Optimization Framework for Versatile GPU Workloads,"PLDI, 2021.

[AI-GPU显存优化领域前沿工作发展史（读博两年里程碑） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/419019170)