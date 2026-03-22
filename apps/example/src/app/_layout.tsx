if (__DEV__) {
  require("../../ReactotronConfig");
}

import { DarkTheme, DefaultTheme, ThemeProvider } from '@react-navigation/native';
import React from 'react';
import { useColorScheme } from 'react-native';
import Reactotron from 'reactotron-react-native';

import AppTabs from '@/components/app-tabs';
import { Slot } from 'expo-router';

function TabLayout() {
  const colorScheme = useColorScheme();
  return (
    <ThemeProvider value={colorScheme === 'dark' ? DarkTheme : DefaultTheme}>
      <Slot />
    </ThemeProvider>
  );
}

export default __DEV__ && Reactotron.overlay
  ? Reactotron.overlay(TabLayout)
  : TabLayout;
