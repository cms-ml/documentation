# CMS-ML FPGA Resource
## Introduction
Welcome to FPGA Resource tab! Our tab is designed to provide accurate, up-to-date, and relevant information about tooling used in CMS for ML on FPGAs.

## FPGA Basics

Field-Programmable Gate Array (FPGA) are  reconfigurable hardware for creating custom digital circuits. FPGAs are build out of Configurable Logic Blocks (CLBs), Programmable Interconnects, and I/O Blocks. They are used in areas where high parallel processing and high performance is needed.

To programm an FPGA one has two main options. The first one is to use a Hardware Design Languages (HDL) while the second one is to use High-Level Design Tools (HLS). In the HDL case one has the control about everything but the design flow can be kind of tricky. Mostly two languages are used VHDL, which is verbose and structured, and Verilog, which is more compact and looks more like a c-style language. In CMS most of the FPGAs are programmed using VHDL.

When it comes to vendors there are two big onces: Xilinx (part of AMD) and Altera (part of Intel). Both vendors provide there own tooling to simulate, synthesis and debug the design. Xilinx FPGAs are used for CMS and they have Vivado for design, simulation, synthesis, and debugging tasks and Vitis for software development for Xilinx FPGAs and SoCs. Intel FPGAs are programmed using Quartus Prime. For HLS tools they come with Vivado HLS (Xilinx) and HLS Compiler (Intel).

To simplify the pipeline from a trained model to an implementation on the FPGA CMS is supporting different tools, which are explained in the inference section ([hls4ml](../../inference/hls4ml.md), [conifer](../../inference/conifer.md), [qonnx](../../inference/qonnx.md)). Furthermore, tools for quantize aware training are used (QKeras, [HGQ](../../training/HGQ.md)).
