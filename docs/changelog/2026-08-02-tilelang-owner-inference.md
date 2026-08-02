# TileLang stable fragment-owner inference

Path C now pins `DatasunriseOU/tilelang` to
`de8bb88cc382b0e78bc804244f79c4be8cc9e75f`, matching `cppmega`. The existing
TVM `e25ca6ae50beee0e907b1e5ed32949879caddde1` and tvm-ffi
`521efeb30bfd9e4946b248b3d76e6391028233a3` pins are unchanged.

The TileLang source includes bodyless Bind-role classification, fail-closed
dependent shared-RMW planning, stable first-use layout source ordering, and
exact fragment-write owner compatibility. H200 promotion remains gated on a
rebuilt immutable wheel/image and the ordered release test.
