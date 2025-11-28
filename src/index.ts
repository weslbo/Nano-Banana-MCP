#!/usr/bin/env node

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  Tool,
  CallToolRequest,
  CallToolResult,
  ErrorCode,
  McpError,
} from "@modelcontextprotocol/sdk/types.js";
import { GoogleGenAI } from "@google/genai";
import { z } from "zod";
import fs from "fs/promises";
import path from "path";
import { config as dotenvConfig } from "dotenv";
import os from "os";

// Load environment variables
dotenvConfig();

const ConfigSchema = z.object({
  geminiApiKey: z.string().min(1, "Gemini API key is required"),
});

type Config = z.infer<typeof ConfigSchema>;

type ImageModel = "gemini-2.5-flash-image-preview" | "gemini-3-pro-image-preview";
type ImageResolution = "1K" | "2K" | "4K";
type AspectRatio = "1:1" | "4:3" | "3:4" | "16:9" | "9:16";

class NanoBananaMCP {
  private server: Server;
  private genAI: GoogleGenAI | null = null;
  private config: Config | null = null;
  private lastImagePath: string | null = null;
  private configSource: 'environment' | 'config_file' | 'not_configured' = 'not_configured';
  private defaultModel: ImageModel = "gemini-2.5-flash-image-preview";

  constructor() {
    this.server = new Server(
      {
        name: "nano-banana-mcp",
        version: "1.0.0",
      },
      {
        capabilities: {
          tools: {},
        },
      }
    );

    this.setupHandlers();
  }

  private setupHandlers() {
    this.server.setRequestHandler(ListToolsRequestSchema, async () => {
      return {
        tools: [
          {
            name: "configure_gemini_token",
            description: `Configure your Gemini API token for nano-banana image generation. This tool stores the API key locally for this session.

🔑 HOW TO GET YOUR API KEY:

1. Visit Google AI Studio: https://aistudio.google.com/apikey
2. Sign in with your Google account
3. Click "Create API Key" or use an existing one
4. Copy the key (starts with "AIza...")

⚠️ WHEN TO USE THIS TOOL:
• Use this ONLY if you don't have GEMINI_API_KEY environment variable set
• This is the fallback method - environment variables are more secure
• The key will be saved to a local config file (.nano-banana-config.json)

🔒 SECURITY BEST PRACTICES:
• PREFERRED: Set GEMINI_API_KEY in your MCP client's environment config
• AVOID: Sharing or committing the config file with your API key
• TIP: Add .nano-banana-config.json to your .gitignore

✅ AFTER CONFIGURATION:
• You can immediately use generate_image, edit_image, and continue_editing
• Use get_configuration_status to verify the setup
• The key persists across sessions via the config file`,
            inputSchema: {
              type: "object",
              properties: {
                apiKey: {
                  type: "string",
                  description: "Your Gemini API key from Google AI Studio (https://aistudio.google.com/apikey). Format: 'AIza...' - a 39-character string starting with 'AIza'.",
                },
              },
              required: ["apiKey"],
            },
          },
          {
            name: "generate_image",
            description: `Generate a NEW image from text prompt. Use this ONLY when creating a completely new image, not when modifying an existing one.

🎨 PROMPT CRAFTING GUIDE FOR BEST RESULTS:

1. STRUCTURE YOUR PROMPT (follow this order):
   • Subject: What is the main focus? (e.g., "a majestic lion", "a cozy coffee shop")
   • Style: Art style or medium (e.g., "oil painting", "photorealistic", "anime style", "watercolor", "3D render")
   • Composition: Framing and perspective (e.g., "close-up portrait", "wide landscape shot", "bird's eye view")
   • Lighting: Light quality and direction (e.g., "golden hour sunlight", "dramatic chiaroscuro", "soft studio lighting")
   • Colors: Color palette or mood (e.g., "vibrant warm colors", "muted earth tones", "high contrast black and white")
   • Details: Specific elements to include (e.g., "intricate patterns", "bokeh background", "rain droplets")

2. BEST PRACTICES:
   • Be specific and descriptive - "elderly woman with silver hair and warm smile" > "old woman"
   • Use artistic references - "in the style of Studio Ghibli" or "reminiscent of Monet's impressionism"
   • Specify quality keywords - "highly detailed", "professional photography", "8K resolution", "masterpiece"
   • Include emotional tone - "serene", "dramatic", "whimsical", "mysterious"
   • Avoid negatives - describe what you WANT, not what you don't want

3. EXAMPLE PROMPTS:
   • Portrait: "A photorealistic portrait of a young astronaut, dramatic rim lighting, helmet reflecting Earth, cinematic composition, 8K detail"
   • Landscape: "Mystical forest at twilight, bioluminescent mushrooms, fog rolling through ancient trees, fantasy art style, ethereal atmosphere"
   • Product: "Minimalist product photography of a luxury perfume bottle, soft gradient background, professional studio lighting, high-end commercial style"

4. MODEL SELECTION:
   • gemini-2.5-flash: Fast generation (~3-5s), good for iterations and drafts
   • gemini-3-pro: Superior quality (~10-15s), better prompt understanding, ideal for final outputs

5. COMMON PITFALLS TO AVOID:
   • ❌ Vague prompts: "a nice picture" → ✅ Specific: "a serene mountain lake at dawn, mist rising, photorealistic"
   • ❌ Negative descriptions: "no people, not dark" → ✅ Positive: "empty scene, bright and airy"
   • ❌ Too many subjects: "cat, dog, bird, fish playing" → ✅ Focus: "a cat playing with a ball of yarn"
   • ❌ Conflicting styles: "realistic anime cartoon" → ✅ Clear style: "Studio Ghibli anime style"

6. OUTPUT INFO:
   • Format: PNG (high quality, lossless)
   • Default save location: ./generated_imgs/ (macOS/Linux) or Documents/nano-banana-images (Windows)
   • Each image gets a unique timestamp filename`,
            inputSchema: {
              type: "object",
              properties: {
                prompt: {
                  type: "string",
                  description: "Detailed text prompt describing the image. Follow the structure: [Subject] + [Style] + [Composition] + [Lighting] + [Colors] + [Details]. Be specific and descriptive for best results.",
                },
                model: {
                  type: "string",
                  enum: ["gemini-2.5-flash-image-preview", "gemini-3-pro-image-preview"],
                  description: "Model selection: 'gemini-2.5-flash-image-preview' for fast iterations (~3-5s) or 'gemini-3-pro-image-preview' for professional quality with advanced reasoning and better prompt understanding (~10-15s). Use flash for drafts, pro for final outputs.",
                },
                resolution: {
                  type: "string",
                  enum: ["1K", "2K", "4K"],
                  description: "Image resolution (gemini-3-pro only). 1K: Fast preview. 2K: Balanced quality/speed. 4K: Maximum detail for print/large displays.",
                },
                aspectRatio: {
                  type: "string",
                  enum: ["1:1", "4:3", "3:4", "16:9", "9:16"],
                  description: "Aspect ratio: 1:1 (social media, icons), 4:3 (standard photo), 3:4 (portrait photo), 16:9 (cinematic, desktop), 9:16 (mobile, stories)",
                },
              },
              required: ["prompt"],
            },
          },
          {
            name: "edit_image",
            description: `Edit a SPECIFIC existing image file, optionally using additional reference images. Use this when you have the exact file path of an image to modify.

✏️ EDIT PROMPT CRAFTING GUIDE:

1. TYPES OF EDITS:
   • Modification: "Change the sky to a dramatic sunset with orange and purple clouds"
   • Addition: "Add a flock of birds flying in the distance"
   • Removal: "Remove the person in the background, fill with natural scenery"
   • Style Transfer: "Transform this photo into a Van Gogh-style oil painting while preserving the composition"
   • Enhancement: "Enhance the lighting to be more dramatic, add subtle lens flare"
   • Color Adjustment: "Change the color palette to warm autumn tones"

2. PROMPT STRUCTURE FOR EDITS:
   • State the action clearly: "Change", "Add", "Remove", "Transform", "Adjust"
   • Specify what to preserve: "Keep the main subject unchanged", "Preserve the overall composition"
   • Describe the desired result: Be specific about the end state

3. USING REFERENCE IMAGES:
   • Style reference: "Apply the artistic style from the reference image to the main image"
   • Element transfer: "Add the object from reference image into the main image"
   • Color reference: "Match the color grading from the reference image"
   • Pose/composition reference: "Adjust the subject's pose to match the reference"

4. BEST PRACTICES:
   • One major edit at a time for best results
   • Be explicit about what should NOT change
   • For complex edits, use continue_editing for iterative refinement

5. EXAMPLE EDIT PROMPTS:
   • "Change the background to a tropical beach at sunset, keep the person exactly as they are"
   • "Add realistic snow falling and frost on the trees, maintain the cozy cabin lighting"
   • "Transform this photo into anime style artwork, preserve facial features and expression"

6. SUPPORTED FILE FORMATS:
   • Input: PNG, JPEG/JPG, WebP
   • Output: Always PNG (high quality, lossless)
   • Max recommended size: 4096x4096 pixels
   • Tip: Larger images may be automatically resized by the model

7. TROUBLESHOOTING:
   • "File not found": Verify the absolute path is correct
   • "Failed to edit": Try simplifying your edit prompt
   • Poor results: Use gemini-3-pro for complex edits
   • Unexpected changes: Be more explicit about what to preserve`,
            inputSchema: {
              type: "object",
              properties: {
                imagePath: {
                  type: "string",
                  description: "Full absolute file path to the main image file to edit (e.g., /Users/name/images/photo.png)",
                },
                prompt: {
                  type: "string",
                  description: "Clear edit instructions: [Action] + [What to change] + [Desired result] + [What to preserve]. Example: 'Change the background to a starry night sky while keeping the subject and foreground unchanged'",
                },
                referenceImages: {
                  type: "array",
                  items: {
                    type: "string"
                  },
                  description: "Optional array of reference image paths for style transfer, element addition, or visual guidance. Mention in your prompt how each reference should be used.",
                },
                model: {
                  type: "string",
                  enum: ["gemini-2.5-flash-image-preview", "gemini-3-pro-image-preview"],
                  description: "Model selection: flash for quick iterations, pro for complex edits requiring better understanding of spatial relationships and style preservation.",
                },
                resolution: {
                  type: "string",
                  enum: ["1K", "2K", "4K"],
                  description: "Output resolution (gemini-3-pro only). Match or exceed original image resolution for best quality.",
                },
                aspectRatio: {
                  type: "string",
                  enum: ["1:1", "4:3", "3:4", "16:9", "9:16"],
                  description: "Output aspect ratio. Usually keep same as original unless intentionally reframing.",
                },
              },
              required: ["imagePath", "prompt"],
            },
          },
          {
            name: "get_configuration_status",
            description: `Check the current Gemini API configuration status and get setup guidance if needed.

📊 WHAT THIS TOOL RETURNS:
• Whether the API token is configured and ready
• The configuration source (environment variable vs config file)
• Security recommendations based on current setup
• Step-by-step setup instructions if not configured

🔍 WHEN TO USE THIS TOOL:
• FIRST: Always check this before attempting image generation if unsure about setup
• TROUBLESHOOTING: When generate_image or edit_image fails with auth errors
• VERIFICATION: After running configure_gemini_token to confirm success
• DEBUGGING: To understand which configuration method is active

💡 CONFIGURATION PRIORITY ORDER:
1. 🥇 Environment variable (GEMINI_API_KEY) - Most secure, recommended
2. 🥈 MCP client env config - Good for per-project setup
3. 🥉 Local config file (.nano-banana-config.json) - Fallback option

🛠️ COMMON ISSUES THIS HELPS DIAGNOSE:
• "API token not configured" errors
• Authentication failures
• Determining if reconfiguration is needed`,
            inputSchema: {
              type: "object",
              properties: {},
              additionalProperties: false,
            },
          },
          {
            name: "continue_editing",
            description: `Continue editing the LAST image that was generated or edited in this session. Use this for iterative improvements and refinements without needing to specify the file path.

🔄 ITERATIVE EDITING WORKFLOW:

1. WHEN TO USE continue_editing:
   • Refining details after initial generation
   • Making incremental adjustments (color, lighting, elements)
   • Fixing specific issues while preserving the rest
   • Building up complexity through multiple passes

2. INCREMENTAL EDIT STRATEGIES:
   • First pass: Get the main composition right
   • Second pass: Refine details and fix issues
   • Third pass: Polish lighting, colors, and atmosphere
   • Final pass: Add finishing touches

3. EFFECTIVE CONTINUATION PROMPTS:
   • Be specific about the change: "Make the eyes more vibrant blue" > "fix the eyes"
   • Reference what works: "Keep the lighting perfect, just adjust the background color"
   • One change at a time for precision: Avoid multiple unrelated edits

4. EXAMPLE CONTINUATION PROMPTS:
   • "The composition is perfect. Now enhance the lighting to be more dramatic with stronger shadows"
   • "Great progress! Add more detail to the texture of the fabric, keep everything else the same"
   • "Almost there - just make the background slightly more blurred for better depth of field"
   • "Love it! Final touch: add a subtle warm color grade to the whole image"

5. TIPS FOR BEST RESULTS:
   • Always acknowledge what's working before requesting changes
   • Use comparative language: "more", "less", "slightly", "much more"
   • If an edit goes wrong, describe what to revert: "Go back to the original sky color"

6. WORKFLOW DECISION GUIDE:
   • Use continue_editing when: Refining the last generated/edited image
   • Use edit_image when: You have a specific file path to edit (not the last image)
   • Use generate_image when: Starting fresh with a new concept

7. SESSION AWARENESS:
   • This tool remembers only the LAST image from the current session
   • If the MCP server restarts, the last image reference is lost
   • Use get_last_image_info to check what image is currently tracked`,
            inputSchema: {
              type: "object",
              properties: {
                prompt: {
                  type: "string",
                  description: "Specific modification to make. Format: [Acknowledge what's good] + [Specific change needed]. Example: 'The subject looks great, now make the background more blurred and add warm golden hour lighting'",
                },
                referenceImages: {
                  type: "array",
                  items: {
                    type: "string"
                  },
                  description: "Optional reference images for this edit iteration. Useful for adding specific elements or matching a particular style.",
                },
                model: {
                  type: "string",
                  enum: ["gemini-2.5-flash-image-preview", "gemini-3-pro-image-preview"],
                  description: "Model selection: Use flash for quick iterations while refining, switch to pro for final polishing passes.",
                },
                resolution: {
                  type: "string",
                  enum: ["1K", "2K", "4K"],
                  description: "Output resolution (gemini-3-pro only). Consider using lower resolution for iterations, higher for final output.",
                },
                aspectRatio: {
                  type: "string",
                  enum: ["1:1", "4:3", "3:4", "16:9", "9:16"],
                  description: "Output aspect ratio. Usually keep consistent across iterations unless intentionally reframing.",
                },
              },
              required: ["prompt"],
            },
          },
          {
            name: "get_last_image_info",
            description: `Get detailed information about the last generated/edited image in this session.

📷 WHAT THIS TOOL RETURNS:
• Full file path to the image
• File size in KB
• Last modified timestamp
• File existence status

🔍 WHEN TO USE THIS TOOL:
• Before continue_editing: Verify which image will be modified
• To share results: Get the exact file path for the user
• After generation: Confirm the image was saved successfully
• Troubleshooting: Check if the file still exists

💡 WORKFLOW TIP:
Call this tool if you're unsure which image continue_editing will modify, especially after multiple generation/edit cycles.`,
            inputSchema: {
              type: "object",
              properties: {},
              additionalProperties: false,
            },
          },
        ] as Tool[],
      };
    });

    this.server.setRequestHandler(CallToolRequestSchema, async (request: CallToolRequest): Promise<CallToolResult> => {
      try {
        switch (request.params.name) {
          case "configure_gemini_token":
            return await this.configureGeminiToken(request);
          
          case "generate_image":
            return await this.generateImage(request);
          
          case "edit_image":
            return await this.editImage(request);
          
          case "get_configuration_status":
            return await this.getConfigurationStatus();
          
          case "continue_editing":
            return await this.continueEditing(request);
          
          case "get_last_image_info":
            return await this.getLastImageInfo();
          
          default:
            throw new McpError(ErrorCode.MethodNotFound, `Unknown tool: ${request.params.name}`);
        }
      } catch (error) {
        if (error instanceof McpError) {
          throw error;
        }
        throw new McpError(ErrorCode.InternalError, `Tool execution failed: ${error instanceof Error ? error.message : String(error)}`);
      }
    });
  }

  private async configureGeminiToken(request: CallToolRequest): Promise<CallToolResult> {
    const { apiKey } = request.params.arguments as { apiKey: string };
    
    try {
      ConfigSchema.parse({ geminiApiKey: apiKey });
      
      this.config = { geminiApiKey: apiKey };
      this.genAI = new GoogleGenAI({ apiKey });
      this.configSource = 'config_file'; // Manual configuration via tool
      
      await this.saveConfig();
      
      return {
        content: [
          {
            type: "text",
            text: "✅ Gemini API token configured successfully! You can now use nano-banana image generation features.",
          },
        ],
      };
    } catch (error) {
      if (error instanceof z.ZodError) {
        throw new McpError(ErrorCode.InvalidParams, `Invalid API key: ${error.errors[0]?.message}`);
      }
      throw error;
    }
  }

  private async generateImage(request: CallToolRequest): Promise<CallToolResult> {
    if (!this.ensureConfigured()) {
      throw new McpError(ErrorCode.InvalidRequest, "Gemini API token not configured. Use configure_gemini_token first.");
    }

    const { prompt, model, resolution, aspectRatio } = request.params.arguments as {
      prompt: string;
      model?: ImageModel;
      resolution?: ImageResolution;
      aspectRatio?: AspectRatio;
    };

    const selectedModel = model || this.defaultModel;
    
    try {
      // Build configuration for Gemini 3 Pro
      const config: any = {
        responseModalities: ['TEXT', 'IMAGE']
      };

      // Add resolution and aspect ratio for Gemini 3 Pro
      if (selectedModel === "gemini-3-pro-image-preview") {
        if (resolution) {
          config.imageSize = resolution;
        }
        if (aspectRatio) {
          config.aspectRatio = aspectRatio;
        }
      }

      const response = await this.genAI!.models.generateContent({
        model: selectedModel,
        contents: prompt,
        config: Object.keys(config).length > 1 ? config : undefined,
      });
      
      // Process response to extract image data
      const content: any[] = [];
      const savedFiles: string[] = [];
      let textContent = "";
      
      // Get appropriate save directory based on OS
      const imagesDir = this.getImagesDirectory();
      
      // Create directory
      await fs.mkdir(imagesDir, { recursive: true, mode: 0o755 });
      
      if (response.candidates && response.candidates[0]?.content?.parts) {
        for (const part of response.candidates[0].content.parts) {
          // Process text content
          if (part.text) {
            textContent += part.text;
          }
          
          // Process image data
          if (part.inlineData?.data) {
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            const randomId = Math.random().toString(36).substring(2, 8);
            const fileName = `generated-${timestamp}-${randomId}.png`;
            const filePath = path.join(imagesDir, fileName);
            
            const imageBuffer = Buffer.from(part.inlineData.data, 'base64');
            await fs.writeFile(filePath, imageBuffer);
            savedFiles.push(filePath);
            this.lastImagePath = filePath;
            
            // Add image to MCP response
            content.push({
              type: "image",
              data: part.inlineData.data,
              mimeType: part.inlineData.mimeType || "image/png",
            });
          }
        }
      }
      
      // Build response content
      const modelName = selectedModel === "gemini-3-pro-image-preview" ? "Gemini 3 Pro Image" : "Gemini 2.5 Flash Image";
      let statusText = `🎨 Image generated with nano-banana (${modelName})!\n\nPrompt: "${prompt}"`;

      if (selectedModel === "gemini-3-pro-image-preview") {
        statusText += `\nModel: Professional quality with advanced reasoning`;
        if (resolution) statusText += `\nResolution: ${resolution}`;
        if (aspectRatio) statusText += `\nAspect Ratio: ${aspectRatio}`;
      }
      
      if (textContent) {
        statusText += `\n\nDescription: ${textContent}`;
      }
      
      if (savedFiles.length > 0) {
        statusText += `\n\n📁 Image saved to:\n${savedFiles.map(f => `- ${f}`).join('\n')}`;
        statusText += `\n\n💡 View the image by:`;
        statusText += `\n1. Opening the file at the path above`;
        statusText += `\n2. Clicking on "Called generate_image" in Cursor to expand the MCP call details`;
        statusText += `\n\n🔄 To modify this image, use: continue_editing`;
        statusText += `\n📋 To check current image info, use: get_last_image_info`;
      } else {
        statusText += `\n\nNote: No image was generated. The model may have returned only text.`;
        statusText += `\n\n💡 Tip: Try running the command again - sometimes the first call needs to warm up the model.`;
      }
      
      // Add text content first
      content.unshift({
        type: "text",
        text: statusText,
      });
      
      return { content };
      
    } catch (error) {
      console.error("Error generating image:", error);
      throw new McpError(
        ErrorCode.InternalError,
        `Failed to generate image: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }

  private async editImage(request: CallToolRequest): Promise<CallToolResult> {
    if (!this.ensureConfigured()) {
      throw new McpError(ErrorCode.InvalidRequest, "Gemini API token not configured. Use configure_gemini_token first.");
    }

    const { imagePath, prompt, referenceImages, model, resolution, aspectRatio } = request.params.arguments as {
      imagePath: string;
      prompt: string;
      referenceImages?: string[];
      model?: ImageModel;
      resolution?: ImageResolution;
      aspectRatio?: AspectRatio;
    };

    const selectedModel = model || this.defaultModel;
    
    try {
      // Prepare the main image
      const imageBuffer = await fs.readFile(imagePath);
      const mimeType = this.getMimeType(imagePath);
      const imageBase64 = imageBuffer.toString('base64');
      
      // Prepare all image parts
      const imageParts: any[] = [
        { 
          inlineData: {
            data: imageBase64,
            mimeType: mimeType,
          }
        }
      ];
      
      // Add reference images if provided
      if (referenceImages && referenceImages.length > 0) {
        for (const refPath of referenceImages) {
          try {
            const refBuffer = await fs.readFile(refPath);
            const refMimeType = this.getMimeType(refPath);
            const refBase64 = refBuffer.toString('base64');
            
            imageParts.push({
              inlineData: {
                data: refBase64,
                mimeType: refMimeType,
              }
            });
          } catch (error) {
            // Continue with other images, don't fail the entire operation
            continue;
          }
        }
      }
      
      // Add the text prompt
      imageParts.push({ text: prompt });

      // Build configuration for Gemini 3 Pro
      const config: any = {
        responseModalities: ['TEXT', 'IMAGE']
      };

      // Add resolution and aspect ratio for Gemini 3 Pro
      if (selectedModel === "gemini-3-pro-image-preview") {
        if (resolution) {
          config.imageSize = resolution;
        }
        if (aspectRatio) {
          config.aspectRatio = aspectRatio;
        }
      }

      // Use new API format with multiple images and text
      const response = await this.genAI!.models.generateContent({
        model: selectedModel,
        contents: [
          {
            parts: imageParts
          }
        ],
        config: Object.keys(config).length > 1 ? config : undefined,
      });
      
      // Process response
      const content: any[] = [];
      const savedFiles: string[] = [];
      let textContent = "";
      
      // Get appropriate save directory
      const imagesDir = this.getImagesDirectory();
      await fs.mkdir(imagesDir, { recursive: true, mode: 0o755 });
      
      // Extract image from response
      if (response.candidates && response.candidates[0]?.content?.parts) {
        for (const part of response.candidates[0].content.parts) {
          if (part.text) {
            textContent += part.text;
          }
          
          if (part.inlineData) {
            // Save edited image
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            const randomId = Math.random().toString(36).substring(2, 8);
            const fileName = `edited-${timestamp}-${randomId}.png`;
            const filePath = path.join(imagesDir, fileName);
            
            if (part.inlineData.data) {
              const imageBuffer = Buffer.from(part.inlineData.data, 'base64');
              await fs.writeFile(filePath, imageBuffer);
              savedFiles.push(filePath);
              this.lastImagePath = filePath;
            }
            
            // Add to MCP response
            if (part.inlineData.data) {
              content.push({
                type: "image",
                data: part.inlineData.data,
                mimeType: part.inlineData.mimeType || "image/png",
              });
            }
          }
        }
      }
      
      // Build response
      const modelName = selectedModel === "gemini-3-pro-image-preview" ? "Gemini 3 Pro Image" : "Gemini 2.5 Flash Image";
      let statusText = `🎨 Image edited with nano-banana (${modelName})!\n\nOriginal: ${imagePath}\nEdit prompt: "${prompt}"`;

      if (selectedModel === "gemini-3-pro-image-preview") {
        statusText += `\nModel: Professional quality with advanced reasoning`;
        if (resolution) statusText += `\nResolution: ${resolution}`;
        if (aspectRatio) statusText += `\nAspect Ratio: ${aspectRatio}`;
      }
      
      if (referenceImages && referenceImages.length > 0) {
        statusText += `\n\nReference images used:\n${referenceImages.map(f => `- ${f}`).join('\n')}`;
      }
      
      if (textContent) {
        statusText += `\n\nDescription: ${textContent}`;
      }
      
      if (savedFiles.length > 0) {
        statusText += `\n\n📁 Edited image saved to:\n${savedFiles.map(f => `- ${f}`).join('\n')}`;
        statusText += `\n\n💡 View the edited image by:`;
        statusText += `\n1. Opening the file at the path above`;
        statusText += `\n2. Clicking on "Called edit_image" in Cursor to expand the MCP call details`;
        statusText += `\n\n🔄 To continue editing, use: continue_editing`;
        statusText += `\n📋 To check current image info, use: get_last_image_info`;
      } else {
        statusText += `\n\nNote: No edited image was generated.`;
        statusText += `\n\n💡 Tip: Try running the command again - sometimes the first call needs to warm up the model.`;
      }
      
      content.unshift({
        type: "text",
        text: statusText,
      });
      
      return { content };
      
    } catch (error) {
      throw new McpError(
        ErrorCode.InternalError,
        `Failed to edit image: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }

  private async getConfigurationStatus(): Promise<CallToolResult> {
    const isConfigured = this.config !== null && this.genAI !== null;
    
    let statusText: string;
    let sourceInfo = "";
    
    if (isConfigured) {
      statusText = "✅ Gemini API token is configured and ready to use";
      
      switch (this.configSource) {
        case 'environment':
          sourceInfo = "\n📍 Source: Environment variable (GEMINI_API_KEY)\n💡 This is the most secure configuration method.";
          break;
        case 'config_file':
          sourceInfo = "\n📍 Source: Local configuration file (.nano-banana-config.json)\n💡 Consider using environment variables for better security.";
          break;
      }
    } else {
      statusText = "❌ Gemini API token is not configured";
      sourceInfo = `

📝 Configuration options (in priority order):
1. 🥇 MCP client environment variables (Recommended)
2. 🥈 System environment variable: GEMINI_API_KEY  
3. 🥉 Use configure_gemini_token tool

💡 For the most secure setup, add this to your MCP configuration:
"env": { "GEMINI_API_KEY": "your-api-key-here" }`;
    }
    
    return {
      content: [
        {
          type: "text",
          text: statusText + sourceInfo,
        },
      ],
    };
  }

  private async continueEditing(request: CallToolRequest): Promise<CallToolResult> {
    if (!this.ensureConfigured()) {
      throw new McpError(ErrorCode.InvalidRequest, "Gemini API token not configured. Use configure_gemini_token first.");
    }

    if (!this.lastImagePath) {
      throw new McpError(ErrorCode.InvalidRequest, "No previous image found. Please generate or edit an image first, then use continue_editing for subsequent edits.");
    }

    const { prompt, referenceImages, model, resolution, aspectRatio } = request.params.arguments as {
      prompt: string;
      referenceImages?: string[];
      model?: ImageModel;
      resolution?: ImageResolution;
      aspectRatio?: AspectRatio;
    };

    // 检查最后的图片文件是否存在
    try {
      await fs.access(this.lastImagePath);
    } catch {
      throw new McpError(ErrorCode.InvalidRequest, `Last image file not found at: ${this.lastImagePath}. Please generate a new image first.`);
    }

    // Use editImage logic with lastImagePath

    return await this.editImage({
      method: "tools/call",
      params: {
        name: "edit_image",
        arguments: {
          imagePath: this.lastImagePath,
          prompt: prompt,
          referenceImages: referenceImages,
          model: model,
          resolution: resolution,
          aspectRatio: aspectRatio,
        }
      }
    } as CallToolRequest);
  }

  private async getLastImageInfo(): Promise<CallToolResult> {
    if (!this.lastImagePath) {
      return {
        content: [
          {
            type: "text",
            text: "📷 No previous image found.\n\nPlease generate or edit an image first, then this command will show information about your last image.",
          },
        ],
      };
    }

    // 检查文件是否存在
    try {
      await fs.access(this.lastImagePath);
      const stats = await fs.stat(this.lastImagePath);
      
      return {
        content: [
          {
            type: "text",
            text: `📷 Last Image Information:\n\nPath: ${this.lastImagePath}\nFile Size: ${Math.round(stats.size / 1024)} KB\nLast Modified: ${stats.mtime.toLocaleString()}\n\n💡 Use continue_editing to make further changes to this image.`,
          },
        ],
      };
    } catch {
      return {
        content: [
          {
            type: "text",
            text: `📷 Last Image Information:\n\nPath: ${this.lastImagePath}\nStatus: ❌ File not found\n\n💡 The image file may have been moved or deleted. Please generate a new image.`,
          },
        ],
      };
    }
  }

  private ensureConfigured(): boolean {
    return this.config !== null && this.genAI !== null;
  }

  private getMimeType(filePath: string): string {
    const ext = path.extname(filePath).toLowerCase();
    switch (ext) {
      case '.jpg':
      case '.jpeg':
        return 'image/jpeg';
      case '.png':
        return 'image/png';
      case '.webp':
        return 'image/webp';
      default:
        return 'image/jpeg';
    }
  }

  private getImagesDirectory(): string {
    const platform = os.platform();
    
    if (platform === 'win32') {
      // Windows: Use Documents folder
      const homeDir = os.homedir();
      return path.join(homeDir, 'Documents', 'nano-banana-images');
    } else {
      // macOS/Linux: Use current directory or home directory if in system paths
      const cwd = process.cwd();
      const homeDir = os.homedir();
      
      // If in system directories, use home directory instead
      if (cwd.startsWith('/usr/') || cwd.startsWith('/opt/') || cwd.startsWith('/var/')) {
        return path.join(homeDir, 'nano-banana-images');
      }
      
      return path.join(cwd, 'generated_imgs');
    }
  }

  private async saveConfig(): Promise<void> {
    if (this.config) {
      const configPath = path.join(process.cwd(), '.nano-banana-config.json');
      await fs.writeFile(configPath, JSON.stringify(this.config, null, 2));
    }
  }

  private async loadConfig(): Promise<void> {
    // Try to load from environment variable first
    const envApiKey = process.env.GEMINI_API_KEY;
    if (envApiKey) {
      try {
        this.config = ConfigSchema.parse({ geminiApiKey: envApiKey });
        this.genAI = new GoogleGenAI({ apiKey: this.config.geminiApiKey });
        this.configSource = 'environment';
        return;
      } catch (error) {
        // Invalid API key in environment
      }
    }
    
    // Fallback to config file
    try {
      const configPath = path.join(process.cwd(), '.nano-banana-config.json');
      const configData = await fs.readFile(configPath, 'utf-8');
      const parsedConfig = JSON.parse(configData);
      
      this.config = ConfigSchema.parse(parsedConfig);
      this.genAI = new GoogleGenAI({ apiKey: this.config.geminiApiKey });
      this.configSource = 'config_file';
    } catch {
      // Config file doesn't exist or is invalid, that's okay
      this.configSource = 'not_configured';
    }
  }

  public async run(): Promise<void> {
    await this.loadConfig();
    
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
  }
}

const server = new NanoBananaMCP();
server.run().catch(console.error);