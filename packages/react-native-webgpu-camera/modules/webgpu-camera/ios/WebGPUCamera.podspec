require 'json'

package = JSON.parse(File.read(File.join(__dir__, '..', '..', '..', 'package.json')))

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

  s.source_files = "**/*.{h,m,mm,swift,hpp,cpp}"
  s.vendored_libraries = 'rust/libwebgpu_camera.a'

  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'SWIFT_INCLUDE_PATHS' => '$(PODS_TARGET_SRCROOT)/rust',
  }
end
