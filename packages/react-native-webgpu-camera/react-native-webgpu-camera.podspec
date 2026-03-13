require 'json'

package = JSON.parse(File.read(File.join(__dir__, 'package.json')))

Pod::Spec.new do |s|
  s.name         = 'react-native-webgpu-camera'
  s.version      = package['version']
  s.summary      = 'WebGPU camera pipeline for React Native'
  s.homepage     = 'https://github.com/AdrianAcala/react-native-webgpu-camera'
  s.license      = { type: 'MIT' }
  s.author       = 'Adrian Acala'
  s.platforms    = { ios: '15.1' }
  s.source       = { git: 'https://github.com/AdrianAcala/react-native-webgpu-camera.git', tag: s.version }
  s.source_files = 'modules/webgpu-camera/ios/**/*.{swift,h,m}'

  s.dependency 'ExpoModulesCore'
end
