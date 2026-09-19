# Personal-small-scale-ML-Compiler

Mali ML kompajler pisan od nule u C++, koji ONNX graf modela provodi kroz graph IR, loop IR i C backend, sa eksplicitnim scheduling primitivama (tiling, fuzija, vektorizacija) za optimizaciju izvršavanja na CPU-u. Projekat uključuje i port dva prolaza u MLIR radi poređenja pristupa, kao i opcioni backend za izmišljeni NPU akcelerator sa scratchpad memorijom.

Ovo je lični projekat u ranoj fazi razvoja, čiji je cilj razumevanje kompajlerske i scheduling logike do najsitnijih detalja, uz javno vidljive korake napretka.