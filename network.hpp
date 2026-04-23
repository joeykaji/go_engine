#pragma once
#include <torch/script.h>
#include <array>
#include <vector>

class GoNetwork {
public:
    GoNetwork(const std::string& path) {
        module = torch::jit::load(path);
        module.eval();
    }

    // input: black[361], white[361], blackMove
    // output: policy[362], value
    std::pair<std::array<float,362>, float>
    inference(const std::array<float,361>& black,
              const std::array<float,361>& white,
              bool blackMove) {

        // build [1, 3, 19, 19] tensor
        auto input = torch::zeros({1, 3, 19, 19});
        for (int i = 0; i < 361; i++) {
            input[0][0][i/19][i%19] = black[i];
            input[0][1][i/19][i%19] = white[i];
            input[0][2][i/19][i%19] = blackMove ? 1.0f : 0.0f;
        }

        std::vector<torch::jit::IValue> inputs = {input};
        auto output = module.forward(inputs).toTuple();

        // policy
        auto policy_tensor = output->elements()[0].toTensor();
        auto policy_softmax = torch::softmax(policy_tensor, 1);
        std::array<float, 362> policy;
        for (int i = 0; i < 362; i++)
            policy[i] = policy_softmax[0][i].item<float>();

        // value
        float value = output->elements()[1].toTensor()[0][0].item<float>();

        return {policy, value};
    }

private:
    torch::jit::script::Module module;
};
