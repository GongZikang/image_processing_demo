def conv(self, pic, kernel, stride, padding， mode='SAME'):
  #卷积核大小
  ks = kernel.shape[:2]
  pad = None
  #卷积模式，输出填充边界
  if mode == 'Full':
      pad = [ks[0] -1, ks[1]-1, ks[0]-1, ks[1]-1]
  elif mode == "VALID":
      pad = [0, 0, 0, 0]
  elif mode == "SAME":
      pad = [(ks[0] - 1) // 2, (ks[1] - 1) // 2,
               (ks[0] - 1) // 2, (ks[1] - 1) // 2]
      if ks[0] % 2 == 0:
          pad[2] += 1
      if ks[1] % 2 == 0:
          pad[3] += 1
  #如果模式不是三种中的一种，输出不可用mode
  else:
      print("Invalid mode")
  padded_inputs = np.pad(inputs, pad_width=((pad[0], pad[2]), (pad[1], pad[3]), (0, 0)), mode="constant")
  #计算填充后的高度和宽度
  height, width, channels = inputs.shape
  out_width = int((width + pad[0] + pad[2] - ks[0]) / stride + 1)
  out_height = int((height + pad[1] + pad[3] - ks[1]) / stride + 1)
  #卷积
  outputs = np.empty(shape=(out_height, out_width))
  for r, y in enumerate(range(0, padded_inputs.shape[0] - ks[1] + 1, stride)):
      for c, x in enumerate(range(0, padded_inputs.shape[1] - ks[0] + 1, stride)):
          outputs[r][c] = np.sum(padded_inputs[y:y + ks[1], x:x + ks[0], :] * kernel)
  return outputs
