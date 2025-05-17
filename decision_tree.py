def component_quality(functionality, temperature, vibration, aging):
    """
    Determines the quality grade of an electronic component based on a series of tests.
    """
    # Initial Condition: Basic Functionality
    if functionality >= 0.28:
        # Second Condition: Temperature Resistance
        if temperature >= 7.2:
            # Third Condition: Vibration Resistance
            if vibration >= 32.1:
                # Fourth Condition: Aging Test
                if aging >= 18.3:
                    return "Grade A (5)"  # Passes all tests
                else:
                    return "Scrap (0)"  # Fails Aging Test
            else:
                return "Scrap (0)"  # Fails Vibration Test
        else:
            return "Scrap (0)"  # Fails Temperature Test
    else:
        return "Scrap (0)"  # Fails Basic Functionality

# Example components
component1 = component_quality(0.3, 8.0, 35.0, 20.0)
print(f"Component 1: {component1}")  # Grade A (5)

component2 = component_quality(0.1, 5.0, 15.0, 10.0)
print(f"Component 2: {component2}")  # Scrap (0)

component3 = component_quality(0.5, 10.0, 20.0, 25.0)
print(f"Component 3: {component3}")  # Scrap (0)

component4 = component_quality(0.4, 9.0, 33.0, 15.0)
print(f"Component 4: {component4}")  # Scrap (0)