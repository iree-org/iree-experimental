configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(CUDNN_TEST_DEPENDS
        FileCheck count not
        cudnn-opt
        )

add_lit_testsuite(check-cudnn "Running the cudnn regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${CUDNN_TEST_DEPENDS}
        )
set_target_properties(check-cudnn PROPERTIES FOLDER "Tests")

add_lit_testsuites(CUDNN ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${CUDNN_TEST_DEPENDS})
