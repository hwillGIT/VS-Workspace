// Documentation: https://docs.anythingllm.com/agent/custom/introduction

class ArticleType {
  static get TYPES() {
    return {
      all_sports: "",
      football: "football",
      cricket: "cricket",
      rugby_union: "rugby-union",
      rugby_league: "rugby-league",
      tennis: "tennis",
      golf: "golf",
      formula1: "formula1",
      motorsport: "motorsport",
      boxing: "boxing",
      athletics: "athletics",
      cycling: "cycling",
      winter_sports: "winter-sports",
      disability_sport: "disability-sport"
    };
  }

  static getName(type) {
    return type.replace(/_/g, " ").replace(/\b\w/g, char => char.toUpperCase());
  }

  static getUrl(type) {
    if (type === "all_sports" || !this.TYPES[type]) return "https://feeds.bbci.co.uk/sport/rss.xml";
    return `https://feeds.bbci.co.uk/sport/${this.TYPES[type]}/rss.xml`;
  }
}

/**
 * @typedef {Object} AnythingLLM
 * @property {import('./plugin.json')} config - your plugin's config
 * @property {function(string|Error): void} logger - Logging function
 * @property {function(string): void} introspect - Print a string to the UI while agent skill is running
 * @property {{getLinkContent: function(url): Promise<{success: boolean, content: string}>}} webScraper - Scrape a website easily to bypass user-agent restrictions.
 */

/** @type {AnythingLLM} */
module.exports.runtime = {
  /**
   * @param {import('./plugin.json')['entrypoint']['params']} args - Arguments passed to the agent skill - defined in plugin.json
   */
  handler: async function (args = {}) {
    const callerId = `Using tool ${this.config.name}-v${this.config.version}`;
    this.introspect(callerId);
    this.logger(callerId);

    try {
      if (args.action === "getBBCSportFeed") return await this.getBBCSportFeed(args.type || "all_sports");
      if (args.action === "getBBCSportContent") return await this.getBBCSportContent(args.url);

      return `Nothing to do`;
    } catch (e) {
      this.logger(e)
      this.introspect(
        `${callerId} failed to execute. Reason: ${e.message}`
      );
      return `Failed to execute agent skill. Error ${e.message}`;
    }
  },

  /**
   * Fetch the latest BBC Sport feed for a given type via RSS.
   * @param {string} type - The type of BBC Sport feed to retrieve.
   */
  getBBCSportFeed: async function (type) {
    try {
      this.logger(`getBBCSportFeed: ${type}`);
      this.introspect(`Starting BBC Sport Feed retrieval for articles in the '${ArticleType.getName(type)}' category...`);

      const url = ArticleType.getUrl(type);
      const response = await fetch(url);
      if (response.status !== 200) throw new Error(`Error: '${type}' (${url}) not found (${response.status})`);

      const xml2js = require('xml2js');
      const result = await xml2js.parseStringPromise(await response.text());
      const items = result.rss.channel[0].item.map(item => ({
        title: item.title[0],
        description: item.description[0],
        link: item.link[0],
        published: item.pubDate[0]
      }));

      const completionDescription = `Retrieved ${items.length} news items from BBC Sport Feed for articles in the '${ArticleType.getName(type)}' category.`;
      this.introspect(completionDescription);
      this.logger('getBBCSportFeed: complete');

      return JSON.stringify(items);
    } catch (err) {
      const errorDescription = `Failed to retrieve any news items from BBC Sport Feed: ${err.message}`;
      this.logger(`getBBCSportFeed: ${errorDescription}`);
      this.introspect(`Error: ${errorDescription}`);
      return `Error: ${errorDescription}`;
    }
  },

  /**
   * Fetch the content of a BBC Sport article from a given URL - just scrapes the page using the webScraper tool
   * that bypasses user-agent restrictions and other traditional `fetch` restrictions.
   * @param {string} url - The full URL of the BBC Sport article to retrieve the content of.
   */
  getBBCSportContent: async function (url = null) {
    try {
      if (!url) throw new Error("No URL provided");
      this.introspect(`Starting BBC Sport Article retrieval from '${url}'...`);
      this.logger(`getBBCSportContent: Retrieving content from '${url}'...`);

      const { success, content } = await this.webScraper.getLinkContent(url);
      if (!success) throw new Error(`Failed to retrieve BBC Sport Article content from '${url}'`);

      this.introspect(`Retrieved BBC Sport Article content from '${url}' (${content.length} characters).`);
      this.logger(`getBBCSportContent: Found content.`);

      return content;
    } catch (err) {
      const errorDescription = `Failed to retrieve BBC Sport Article content from '${url}': ${err.message}`;
      this.introspect(errorDescription);
      this.logger(`getBBCSportContent: ${errorDescription}`);
      return `Error: ${errorDescription}`;
    }
  }
};
