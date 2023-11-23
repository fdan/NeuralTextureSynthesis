#include <torch/extension.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


at::Tensor computeHistogram(at::Tensor const &t, unsigned int numBins)
{
  at::Tensor unsqueezed(t);

  if (unsqueezed.ndimension() == 1)
    unsqueezed.unsqueeze_(0);

  if (unsqueezed.ndimension() > 2)
    unsqueezed = unsqueezed.view({unsqueezed.size(0), -1});

  unsigned int c = unsqueezed.size(0);     // Number of channels
  unsigned int n = unsqueezed.numel() / c; // Number of element per channel

  at::Tensor min = torch::amin(unsqueezed, 1, true);
  at::Tensor max = torch::amax(unsqueezed, 1, true);

  at::Tensor h = at::zeros({int(c), int(numBins)}, unsqueezed.type())

  // need to loop through channels?


//  computeHistogram<<<(c*n) / THREAD_COUNT + 1, THREAD_COUNT>>>(unsqueezed.data<float>(),
//    							       h.data<float>(),
//    							       min.data<float>(),
//    							       max.data<float>(),
//    							       c, n, numBins);

  return h;
}