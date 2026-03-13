require 'json'

package = JSON.parse(File.read(File.join(__dir__, '..', '..', '..', 'package.json')))

# Resolve react-native-skia cpp root for Dawn/Skia headers
skia_pkg_root = File.expand_path('../../../../../packages/react-native-skia/packages/skia', __dir__)

Pod::Spec.new do |s|
  s.name           = 'WebGPUCamera'
  s.version        = package['version']
  s.summary        = 'WebGPU camera pipeline for React Native'
  s.description    = 'WebGPU camera pipeline for React Native'
  s.license        = 'MIT'
  s.author         = 'Adrian Acala'
  s.homepage       = 'https://github.com/AdrianAcala/react-native-webgpu-camera'
  s.platforms      = {
    :ios => '15.1'
  }
  s.swift_version  = '5.9'
  s.source         = { git: 'https://github.com/AdrianAcala/react-native-webgpu-camera.git' }
  s.static_framework = true

  s.dependency 'ExpoModulesCore'
  s.dependency 'react-native-skia'

  s.source_files = "**/*.{h,m,mm,swift,hpp,cpp}"
  s.vendored_libraries = 'rust/libwebgpu_camera.a'

  s.frameworks = ['CoreVideo', 'IOSurface', 'MetalKit']

  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'CLANG_CXX_LANGUAGE_STANDARD' => 'c++17',
    'GCC_PREPROCESSOR_DEFINITIONS' => '$(inherited) SK_GRAPHITE=1 RCT_NEW_ARCH_ENABLED=1',
    'SWIFT_INCLUDE_PATHS' => '$(PODS_TARGET_SRCROOT)/rust',
    'HEADER_SEARCH_PATHS' => [
      '"$(PODS_TARGET_SRCROOT)"',
      "\"#{skia_pkg_root}/cpp/\"/**",
      "\"#{skia_pkg_root}/cpp\"",
      "\"#{skia_pkg_root}/cpp/skia\"",
      "\"#{skia_pkg_root}/cpp/rnskia\"",
      "\"#{skia_pkg_root}/cpp/rnwgpu\"",
      "\"#{skia_pkg_root}/cpp/rnwgpu/api\"",
      "\"#{skia_pkg_root}/cpp/rnwgpu/api/descriptors\"",
      "\"#{skia_pkg_root}/cpp/rnwgpu/async\"",
      "\"#{skia_pkg_root}/cpp/jsi2\"",
      "\"#{skia_pkg_root}/cpp/dawn/include\"",
      "\"#{skia_pkg_root}/apple\"",
      "\"#{skia_pkg_root}/cpp/api\"",
    ].join(' '),
  }
end
