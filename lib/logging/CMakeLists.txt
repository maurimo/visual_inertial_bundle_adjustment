# Minimal logging/check macros (replacing VRS' logging)
add_library(viba_logging INTERFACE)
target_include_directories(viba_logging INTERFACE "../")
target_link_libraries(viba_logging INTERFACE fmt::fmt)
