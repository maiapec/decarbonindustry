import pandas as pd

df = pd.DataFrame(columns=['Component', 'Parameter', 'Value'])

def setDefaultValue(component, parameter):
    ''''
    Returns the value of a parameter for a given component.
    Inputs:
        - component: str, name of the component
        - parameter: str, name of the parameter
    '''
    return df.loc[['Component'] == component & df['Parameter'] == parameter]['Value'].values