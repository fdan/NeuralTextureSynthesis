#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define THREAD_COUNT 1024

__global__ void computeHistogram(float *tensor, float *histogram, float *minv,
                                 float *maxv, unsigned int channels,
                                 unsigned int tensorSize, unsigned int nBins)
{
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < channels * tensorSize)
    {
      // Compute which channel we're in
      unsigned int channel = index / tensorSize;
      // Normalize the value in range [0, numBins]
      float value = (tensor[index] - minv[channel]) / (maxv[channel] - minv[channel]) * float(nBins);
      // Compute bin index
      int bin = min((unsigned int)(value), nBins - 1);
      // Increment relevant bin
      atomicAdd(histogram + (channel * nBins) + bin, 1);
    }
}

__global__ void computeHistogramMasked(float *tensor, float *mask, float *histogram, float *minv,
                                       float *maxv, unsigned int channels, unsigned int tensorSize,
                                       unsigned int nBins) {
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    int masked = 0;
    int unmasked = 0;
    // range check - channels * tensorSize is the whole image
    if (index < channels * tensorSize)
    {
        // Compute which channel we're in
        unsigned int channel = index / tensorSize;
        // Normalize the value in range [0, numBins]
        float value = (tensor[index] - minv[channel]) / (maxv[channel] - minv[channel])* float(nBins);
        // get the mask value
        float maskValue = mask[index];
        if (maskValue != 0.f)
        {
            // Compute bin index
            int bin = min((unsigned int) (value), nBins - 1);
            // Increment relevant bin
            atomicAdd(histogram + (channel * nBins) + bin, 1);
        }
    }
}

// return cummulative histogram shifed to the right by 1
// ==> histogram[c][0] alweays == 0
__global__ void accumulateHistogram(float *histogram, unsigned int nBins)
{
  float t = 0;
  for (unsigned int i=0 ; i < nBins ; ++i)
    {
      float swap = histogram[i + blockIdx.x * nBins];
      histogram[i + blockIdx.x * nBins ] = t;
      t += swap;
    }
}

__global__ void buildSortedLinkmap(float *tensor, unsigned int *linkMap, float *cumulativeHistogram,
                                   unsigned int *localIndexes, long *indirection, float *minv,
                                   float *maxv, unsigned int channels, unsigned int tensorSize, unsigned int nBins)
{
  unsigned int index = threadIdx.x + blockIdx.x* blockDim.x;
  if (index < channels * tensorSize)
    {
      // Shuffle image -- Avoid the blurry top bug
      index = indirection[index];
      // Compute which channel we're in
      unsigned int channel = index / tensorSize;
      // Normalize the value in range [0, numBins]
      float value = (tensor[index] - minv[channel]) / (maxv[channel] - minv[channel]) * float(nBins);
      // Compute bin index
      int binIndex = min((unsigned int)(value), nBins - 1);
      // Increment and retrieve the number of pixel in said bin
      int localIndex = atomicAdd(&localIndexes[(channel * 256) + binIndex], 1);
      // Retrieve the number of pixel in all bin lower (in cummulative histogram)
      unsigned int lowerPixelCount = cumulativeHistogram[(channel * 256) + binIndex];
      // Set the linkmap for indes to it's position as "pseudo-sorted"
      linkMap[index] = lowerPixelCount + localIndex;
    }
}


__global__ void buildSortedLinkmapMasked(float *tensor, float *mask, unsigned int *linkMap,
                                         float *cumulativeHistogram, unsigned int *localIndexes, long *indirection,
                                         float *minv, float *maxv, unsigned int channels, unsigned int tensorSize,
                                         unsigned int nBins)
{
    unsigned int index = threadIdx.x + blockIdx.x* blockDim.x;
    if (index < channels * tensorSize)
    {
        // Shuffle image -- Avoid the blurry top bug
        index = indirection[index];
        // Compute which channel we're in
        unsigned int channel = index / tensorSize;
        // Normalize the value in range [0, numBins]
        float value = (tensor[index] - minv[channel]) / (maxv[channel] - minv[channel]) * float(nBins);
        // get the mask value
        float maskValue = mask[index];
        // Compute bin index
        int binIndex = min((unsigned int)(value), nBins - 1);

        if (maskValue != 0.f)
        {
            // Increment and retrieve the number of pixel in said bin
            int localIndex = atomicAdd(&localIndexes[(channel * 256) + binIndex], 1);
            // Retrieve the number of pixel in all bin lower (in cummulative histogram)
            unsigned int lowerPixelCount = cumulativeHistogram[(channel * 256) + binIndex];
            // Set the linkmap for indes to it's position as "pseudo-sorted"
            linkMap[index] = lowerPixelCount + localIndex;
        }
    }
}


__global__ void rebuild(float *tensor, unsigned int *linkMap, float *targetHistogram,
                        float scale, unsigned int channels, unsigned int tensorSize)
{
  unsigned int index = threadIdx.x + blockIdx.x* blockDim.x;
  if (index < channels * tensorSize)
    {
      unsigned int channel = index / tensorSize;
      unsigned int value = 0;
      for (int i=0 ; i < 256 ; ++i)
	if (linkMap[index] >= targetHistogram[(channel * 256) + i] * scale) value = i;
      tensor[index] = (float)value;
    }
}


__global__ void rebuildMasked(float *tensor, float *mask, unsigned int *linkMap,
                              float *targetHistogram, float scale, unsigned int channels,
                              unsigned int tensorSize, unsigned int nBins)
{
    unsigned int index = threadIdx.x + blockIdx.x* blockDim.x;
    if (index < channels * tensorSize)
    {
        unsigned int channel = index / tensorSize;
        unsigned int value = 0;
        float maskValue = mask[index];
        if (maskValue != 0.f)
        {
            for (int i = 0; i < nBins; ++i)
                if (linkMap[index] >= targetHistogram[(channel * nBins) + i] * scale)
                {
                    value = i;
                }
            tensor[index] = (float) value;
        }
    }
}


__global__ void maskedMin(float *tensor, float *mask, float nonZero, unsigned int channels,
                          unsigned int tensorSize, unsigned int nBins)
{
    unsigned int index = threadIdx.x + blockIdx.x* blockDim.x;
    if (index < channels * tensorSize)
    {
        float maskValue = mask[index];
        if (maskValue != 0.f) {
            float value = (tensor[index] - minv[channel]) / (maxv[channel] - minv[channel]) * float(nBins);
            nonZero[index] = value;
        }
    }
}


__global__ void maskedDiv(float *tensor, float *mask, unsigned int nBins, unsigned int channels,
                          unsigned int tensorSize)
{
    unsigned int index = threadIdx.x + blockIdx.x* blockDim.x;
    if (index < channels * tensorSize)
    {
        float maskValue = mask[index];
        if (maskValue != 0.f) {
            tensor[index] = tensor[index] / float(nBins);
        }
    }
}


at::Tensor computeHistogramMasked(at::Tensor const &t, at::Tensor const &m, unsigned int numBins)
{
  at::Tensor unsqueezed(t);
  unsqueezed = unsqueezed.cuda();
  if (unsqueezed.ndimension() == 1)
    unsqueezed.unsqueeze_(0);
  if (unsqueezed.ndimension() > 2)
    unsqueezed = unsqueezed.view({unsqueezed.size(0), -1});
  
  unsigned int c = unsqueezed.size(0);     // Number of channels
  unsigned int n = unsqueezed.numel() / c; // Number of element per channel

  at::Tensor min = torch::amin(unsqueezed, 1, true).cuda();
  at::Tensor max = torch::amax(unsqueezed, 1, true).cuda();

  at::Tensor unsqueezedMask(m);
  unsqueezedMask = unsqueezedMask.cuda();
  if (unsqueezedMask.ndimension() == 1)
    unsqueezedMask.unsqueeze_(0);
  if (unsqueezedMask.ndimension() > 2)
    unsqueezedMask = unsqueezedMask.view({unsqueezedMask.size(0), -1});

  unsigned int mc = unsqueezed.size(0);     // Number of channels
  unsigned int mn = unsqueezed.numel() / c; // Number of element per channel

  at::Tensor h = at::zeros({int(c), int(numBins)}, unsqueezed.type()).cuda();

  computeHistogramMasked<<<(c*n) / THREAD_COUNT + 1, THREAD_COUNT>>>(
          unsqueezed.data_ptr<float>(),
          unsqueezedMask.data_ptr<float>(),
          h.data_ptr<float>(),
          min.data_ptr<float>(),
          max.data_ptr<float>(),
          c,
          n,
          numBins
  );
  return h;
}

at::Tensor computeHistogram(at::Tensor const &t, unsigned int numBins)
{
  at::Tensor unsqueezed(t);
  unsqueezed = unsqueezed.cuda();

  if (unsqueezed.ndimension() == 1)
    unsqueezed.unsqueeze_(0);
  if (unsqueezed.ndimension() > 2)
    unsqueezed = unsqueezed.view({unsqueezed.size(0), -1});
  
  unsigned int c = unsqueezed.size(0);     // Number of channels
  unsigned int n = unsqueezed.numel() / c; // Number of element per channel
  at::Tensor min = torch::amin(unsqueezed, 1, true).cuda();
  at::Tensor max = torch::amax(unsqueezed, 1, true).cuda();

//  std::cout << "numel: " << unsqueezed.numel() << std::endl;
//  std::cout << "channels: " << c << std::endl;
//  std::cout << "elems per channel: " << n << std::endl;
//  std::cout << "min: " << min.data<float>() << " max: " << max.data<float>() << std::endl;

  at::Tensor h = at::zeros({int(c), int(numBins)}, unsqueezed.type()).cuda();

  computeHistogram<<<(c*n) / THREAD_COUNT + 1, THREAD_COUNT>>>(
          unsqueezed.data_ptr<float>(),
          h.data_ptr<float>(),
          min.data_ptr<float>(),
          max.data_ptr<float>(),
          c,
          n,
          numBins
  );

  return h;
}


void matchHistogram(at::Tensor &featureMaps, at::Tensor &targetHistogram)
{
  static std::map<unsigned int, at::Tensor> randomIndices;

  if (randomIndices[featureMaps.numel()].numel() != featureMaps.numel())
  {
      randomIndices[featureMaps.numel()] = torch::randperm(
              featureMaps.numel(), torch::TensorOptions().dtype(at::kLong)
      ).cuda();
  }

  at::Tensor unsqueezed(featureMaps);
  if (unsqueezed.ndimension() == 1)
    unsqueezed.unsqueeze_(0);
  if (unsqueezed.ndimension() > 2)
    unsqueezed = unsqueezed.view({unsqueezed.size(0), -1});

  unsigned int nBins = targetHistogram.size(1);
  unsigned int c = unsqueezed.size(0);     // Number of channels
  unsigned int n = unsqueezed.numel() / c; // Number of element per channel

  float scale = float(featureMaps.numel()) / targetHistogram.sum().item<float>();
 
  at::Tensor featuresHistogram = computeHistogram(unsqueezed, nBins);
  accumulateHistogram<<<c, 1>>>(featuresHistogram.data_ptr<float>(), nBins);
  accumulateHistogram<<<c, 1>>>(targetHistogram.data_ptr<float>(), nBins);

  unsigned int *linkMap = NULL;
  cudaMalloc(&linkMap, c * n * sizeof(unsigned int));

  unsigned int *localIndexes = NULL;
  cudaMalloc(&localIndexes, c * nBins * sizeof(unsigned int));
  cudaMemset(localIndexes, 0, c * nBins * sizeof(unsigned int));

  at::Tensor min = torch::amin(unsqueezed, 1, true).cuda();
  at::Tensor max = torch::amax(unsqueezed, 1, true).cuda();

  buildSortedLinkmap<<<(c*n) / THREAD_COUNT + 1, THREAD_COUNT>>>(
          featureMaps.data_ptr<float>(),
          linkMap,
          featuresHistogram.data_ptr<float>(),
          localIndexes,
          randomIndices[featureMaps.numel()].data_ptr<long>(),
          min.data_ptr<float>(),
          max.data_ptr<float>(),
          c,
          n,
          nBins
  );

  rebuild<<<(c*n) / THREAD_COUNT + 1, THREAD_COUNT>>>(
          featureMaps.data_ptr<float>(),
          linkMap,
          targetHistogram.data_ptr<float>(),
          scale,
          c,
          n
  );

  featureMaps.div_(float(nBins));
  
  cudaFree(linkMap);
  cudaFree(localIndexes);
}

void matchHistogramMasked(at::Tensor &featureMaps, at::Tensor &mask,
      at::Tensor &targetHistogram)
{
//  auto x = torch::ones({1});
  static std::map<unsigned int, at::Tensor> randomIndices;

  if (randomIndices[featureMaps.numel()].numel() != featureMaps.numel())
  {
      randomIndices[featureMaps.numel()] = torch::randperm(
              featureMaps.numel(), torch::TensorOptions().dtype(at::kLong)
      ).cuda();
  }

  at::Tensor unsqueezed(featureMaps);
  if (unsqueezed.ndimension() == 1)
    unsqueezed.unsqueeze_(0);
  if (unsqueezed.ndimension() > 2)
    unsqueezed = unsqueezed.view({unsqueezed.size(0), -1});

  at::Tensor unsqueezedMask(mask);
  unsqueezedMask = unsqueezedMask.cuda();
  if (unsqueezedMask.ndimension() == 1)
    unsqueezedMask.unsqueeze_(0);
  if (unsqueezedMask.ndimension() > 2)
    unsqueezedMask = unsqueezedMask.view({unsqueezedMask.size(0), -1});

  unsigned int nBins = targetHistogram.size(1);
  unsigned int c = unsqueezed.size(0);     // Number of channels
  unsigned int n = unsqueezed.numel() / c; // Number of element per channel

// targetHistogram.sum() is the sum of counts, i.e. how many elements does the hist count
// scale is therefore a ratio of the number of elements in the feature maps
// to the number of elements in the histogram.
  float maskNonzeroCount = mask.count_nonzero().item<float>();

//  float scale = mask.count_nonzero().item<float>() / targetHistogram.sum().item<float>();
  float scale = maskNonzeroCount / targetHistogram.sum().item<float>();

  at::Tensor featuresHistogram = computeHistogramMasked(
    unsqueezed, unsqueezedMask, nBins
  );

  accumulateHistogram<<<c, 1>>>(featuresHistogram.data_ptr<float>(), nBins);
  accumulateHistogram<<<c, 1>>>(targetHistogram.data_ptr<float>(), nBins);

  unsigned int *linkMap = NULL;
  cudaMalloc(&linkMap, c * n * sizeof(unsigned int));

  unsigned int *localIndexes = NULL;
  cudaMalloc(&localIndexes, c * nBins * sizeof(unsigned int));
  cudaMemset(localIndexes, 0, c * nBins * sizeof(unsigned int));

  // todo: take min and max only where mask equals nonzero
//  unsigned int *nonZero = NULL;
//  cudaMalloc(&nonZero, maskNonzeroCount * sizeof(unsigned int));
//  maskedMin<<<(c*n) / THREAD_COUNT + 1, THREAD_COUNT>>>(
//      unsqueezed.data_ptr<float>(),
//      unsqueezedMask.data_ptr<float>(),
//      nonZero.data_ptr<float>(),
//      c,
//      n,
//      nBins
//  );
//  at::Tensor min = torch::amin(nonZero, 1, true).cuda();
//  at::Tensor max = torch::amax(nonZero, 1, true).cuda();
  at::Tensor min = torch::amin(unsqueezed, 1, true).cuda();
  at::Tensor max = torch::amax(unsqueezed, 1, true).cuda();

  // this doesn't seem to be necessary, but keep for now
  buildSortedLinkmapMasked<<<(c*n) / THREAD_COUNT + 1, THREAD_COUNT>>>(
      featureMaps.data_ptr<float>(),
      mask.data_ptr<float>(),
      linkMap,
      featuresHistogram.data_ptr<float>(),
      localIndexes,
      randomIndices[featureMaps.numel()].data_ptr<long>(),
      min.data_ptr<float>(),
      max.data_ptr<float>(),
      c,
      n,
      nBins
  );

  rebuildMasked<<<(c*n) / THREAD_COUNT + 1, THREAD_COUNT>>>(
      featureMaps.data_ptr<float>(),
      mask.data_ptr<float>(),
      linkMap,
      targetHistogram.data_ptr<float>(),
      scale,
      c,
      n,
      nBins
  );

// this doesn't seem to be necessary, but keep for now
  maskedDiv<<<(c*n) / THREAD_COUNT + 1, THREAD_COUNT>>>(
          featureMaps.data_ptr<float>(),
          mask.data_ptr<float>(),
          nBins,
          c,
          n
  );

  cudaFree(linkMap);
  cudaFree(localIndexes);
}
