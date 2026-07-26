import React, { createContext, useContext, useState } from 'react';

export type EducationLevel = 1 | 2 | 3;

interface EducationContextType {
  level: EducationLevel;
  setLevel: (level: EducationLevel) => void;
  isOpen: boolean;
  setIsOpen: (open: boolean) => void;
  toggleDrawer: () => void;
}

const EducationContext = createContext<EducationContextType | undefined>(undefined);

export const EducationProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [level, setLevel] = useState<EducationLevel>(1);
  const [isOpen, setIsOpen] = useState<boolean>(false);

  const toggleDrawer = () => setIsOpen((prev) => !prev);

  return (
    <EducationContext.Provider value={{ level, setLevel, isOpen, setIsOpen, toggleDrawer }}>
      {children}
    </EducationContext.Provider>
  );
};

export const useEducation = (): EducationContextType => {
  const context = useContext(EducationContext);
  if (!context) {
    throw new Error('useEducation must be used within an EducationProvider');
  }
  return context;
};
