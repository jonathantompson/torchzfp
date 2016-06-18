package = "torchzfp"
version = "1.0-1"
source = {
   url = "git://github.com/jonathantompson/torchzfp.git"
}
description = {
   summary = "A utility library for zfp compression / decompression of tensors",
   detailed = [[
   ]],
   homepage = "...",
   license = "None"
}
dependencies = {
}

build = {
   type = "command",
   build_command = [[
   cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE)
   ]],
   install_command = "cd build && $(MAKE) install"
}
