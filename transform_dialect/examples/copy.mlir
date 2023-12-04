!I_t = tensor<${M}xf32>

func.func private @copy_1d_static(%I : !I_t, %O : !I_t) -> !I_t {
  %0 = linalg.copy ins(%I : !I_t) outs(%O : !I_t) -> !I_t
  return %0 : !I_t
}

!I_dyn_t = tensor<?xf32>

func.func private @copy_1d_dynamic(%I : !I_dyn_t, %O : !I_dyn_t) -> !I_dyn_t {
  %0 = linalg.copy ins(%I : !I_dyn_t) outs(%O : !I_dyn_t) -> !I_dyn_t
  return %0 : !I_dyn_t
}
