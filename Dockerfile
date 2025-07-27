FROM mattermost/mattermost-team-edition:latest

# Switch to root to make changes
USER root

# Set timezone environment variables
ENV TZ=America/New_York
ENV NODE_TZ=America/New_York

# Install required tools
RUN apt-get update && apt-get install -y sed && rm -rf /var/lib/apt/lists/*

# Create entry script
COPY docker-entry.sh /
RUN chmod +x /docker-entry.sh && \
    chown mattermost:mattermost /docker-entry.sh

# Switch back to mattermost user
USER mattermost

ENTRYPOINT ["/docker-entry.sh"]
CMD ["mattermost"] 