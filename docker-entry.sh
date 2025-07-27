#!/bin/bash
set -x

echo "Starting Mattermost..."
echo "Configuring plugin timezones..."

# Configure timezone settings in config.json if it exists
if [ -f /mattermost/config/config.json ]; then
    # Update timezone settings
    sed -i 's/"timezone": "[^"]*"/"timezone": "America\/New_York"/g' /mattermost/config/config.json
    sed -i 's/"defaultTimezone": "[^"]*"/"defaultTimezone": "America\/New_York"/g' /mattermost/config/config.json
    sed -i 's/"location": "[^"]*"/"location": "America\/New_York"/g' /mattermost/config/config.json
    
    # Also update the plugin settings specifically
    sed -i '/"com.github.scottleedavis.mattermost-plugin-remind": {/,/}/ s/"timezone": "[^"]*"/"timezone": "America\/New_York"/g' /mattermost/config/config.json
    sed -i '/"com.github.scottleedavis.mattermost-plugin-remind": {/,/}/ s/"defaultTimezone": "[^"]*"/"defaultTimezone": "America\/New_York"/g' /mattermost/config/config.json
    sed -i '/"com.github.scottleedavis.mattermost-plugin-remind": {/,/}/ s/"location": "[^"]*"/"location": "America\/New_York"/g' /mattermost/config/config.json
    
    # Add force local time setting if not present
    if ! grep -q '"forceLocalTime": true' /mattermost/config/config.json; then
        sed -i '/"com.github.scottleedavis.mattermost-plugin-remind": {/a \                "forceLocalTime": true,' /mattermost/config/config.json
    fi
fi

# Start Mattermost
exec /entrypoint.sh "$@" 